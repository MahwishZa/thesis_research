"""The question builder of the Alzheimer's-specific set, end to end with a fake PubMed and fake models. No network, no model."""

import contextlib
import io
import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path

from experiments.adkqa import build as B
from experiments.adkqa import records as R
from experiments.adkqa import spec
from experiments.medchange.pubmed_asof import EUtils

TOPICS = [("donepezil", "cognition", "SUPPORTED", "Donepezil significantly improved cognition."),
          ("memantine", "agitation", "REFUTED", "There was no significant difference in agitation."),
          ("exercise", "memory", "NOT ENOUGH INFORMATION", "The evidence is insufficient to draw conclusions."),
          ("plasma p-tau217", "amyloid pathology", "SUPPORTED", "Plasma p-tau217 showed high diagnostic accuracy.")]


def article(pmid, x, y, concl, label="CONCLUSIONS", unstructured=False):
    if unstructured:
        body = f"<AbstractText>We reviewed {x} and {y} in dementia. {concl}</AbstractText>"
    else:
        body = (f'<AbstractText Label="OBJECTIVE">To assess {x} and {y}.</AbstractText>'
                f'<AbstractText Label="{label}">{concl}</AbstractText>')
    return (f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><ArticleTitle>{x} and {y}</ArticleTitle>"
            f"<Abstract>{body}</Abstract><PublicationTypeList><PublicationType>Meta-Analysis</PublicationType>"
            f"</PublicationTypeList></Article><MeshHeadingList><MeshHeading><DescriptorName MajorTopicYN=\"Y\">Alzheimer Disease"
            f"</DescriptorName></MeshHeading><MeshHeading><DescriptorName MajorTopicYN=\"Y\">{x.title()}</DescriptorName>"
            f"</MeshHeading></MeshHeadingList></MedlineCitation></PubmedArticle>")


def universe(n=200):
    arts = {}
    for i in range(n):
        x, y, lab, concl = TOPICS[i % 4]
        arts[str(500000 + i)] = (f"{x}{i}", y, lab, concl)
    return arts


def fake_eu(arts, search_count=30):
    def opener(url):
        q = urllib.parse.urlparse(url)
        params = dict(urllib.parse.parse_qsl(q.query))
        if "efetch" in q.path:
            ids = params["id"].split(",")
            return ("<PubmedArticleSet>" + "".join(article(p, arts[p][0], arts[p][1], arts[p][3]) for p in ids) +
                    "</PubmedArticleSet>").encode()
        if "esummary" in q.path:
            ids = params["id"].split(",")
            res = {"uids": ids}
            res.update({p: {"sortpubdate": "2024/05/01 00:00", "pubdate": "2024 May", "epubdate": "", "pubtype": [], "source": "J"}
                        for p in ids})
            return json.dumps({"result": res}).encode()
        if "esearch" in q.path:
            ids = [str(500000 + k) for k in range(search_count)]
            return json.dumps({"esearchresult": {"count": str(len(ids)), "idlist": ids}}).encode()
        raise AssertionError(url)
    return EUtils(opener=opener, sleep=lambda s: None)


def counts_file(arts, tmp):
    ids = sorted(arts)
    covered = ("treatment", "diagnosis", "causes_risk", "symptoms")
    per_area = {a: {"core": True, "distinct": 100 if a in covered else 3, "ids": []} for a in
                ("treatment", "prevention", "diagnosis", "causes_risk", "progression", "symptoms", "care_management")}
    for k, pmid in enumerate(ids):
        per_area[covered[k % 4]]["ids"].append(pmid)
    return {"total": 900, "per_area": per_area}


def scripted_generator(arts, wrong_verifier=False):
    def gen(system, user):
        if "Write ONE question" in user:
            title = user.split("TITLE: ", 1)[1].split("\n", 1)[0]
            x = title.split(" and ")[0]
            y = title.split(" and ")[1]
            verdict = next(t[2] for t in TOPICS if x.startswith(t[0]))
            return f"TEMPLATE: effect\nX: {x}\nY: {y}\nVERDICT: {verdict}"
        label = next(t[2] for t in TOPICS if t[3] in user)
        if wrong_verifier:
            label = "REFUTED" if label != "REFUTED" else "SUPPORTED"
        return f"LABEL: {label}"
    return gen


class ParsingTests(unittest.TestCase):
    def test_efetch_xml_is_parsed_into_sections_and_major_descriptors(self):
        rec = R.parse_efetch(("<PubmedArticleSet>" + article("1", "donepezil", "cognition", "It was effective.") +
                              "</PubmedArticleSet>").encode())[0]
        self.assertEqual(rec["sections"][1], ("CONCLUSIONS", "It was effective."))
        self.assertEqual(rec["major_mesh"], ["Alzheimer Disease", "Donepezil"])
        self.assertEqual(spec.cluster_key(rec["major_mesh"], "1"), "donepezil")

    def test_conclusion_and_objectives_follow_the_protocol_rules(self):
        rec = {"sections": [("OBJECTIVE", "To assess it."), ("CONCLUSIONS", "It worked.")]}
        s = R.structure(rec)
        self.assertEqual(s["conclusion"], "It worked.")
        self.assertEqual(s["objectives"], "To assess it.")
        a, b = s["conclusion_span"]
        self.assertEqual(s["abstract"][a:b], "It worked.")
        self.assertEqual(R.structure({"sections": [("AUTHORS\u2019 CONCLUSIONS", "It worked.")]})["conclusion"], "It worked.")
        self.assertEqual(R.structure({"sections": [("Conclusions and relevance", "It worked.")]})["conclusion"], "It worked.")
        tail = R.structure({"sections": [("", "One. Two. Three. Four. Further research is needed.")]})
        self.assertEqual((tail["conclusion"], tail["conclusion_basis"]), ("Three. Four. Further research is needed.", "abstract_tail"))
        self.assertEqual(R.structure({"sections": [("", "We looked. It worked.")]})["conclusion"], "We looked. It worked.")
        disc = R.structure({"sections": [("RESULTS", "r."), ("DISCUSSION", "A. B. C. D.")]})
        self.assertEqual((disc["conclusion"], disc["conclusion_basis"]), ("B. C. D.", "discussion_tail"))
        self.assertEqual(R.structure({"sections": [("DISCUSSION", "A."), ("CONCLUSIONS", "Z.")]})["conclusion_basis"], "labelled")
        self.assertIsNone(R.structure({"sections": [("BACKGROUND", "x"), ("RESULTS", "y")]})["conclusion"])


class DiagnoseTests(unittest.TestCase):
    def test_diagnosis_counts_labels_and_clusters_without_text(self):
        rows = [{"pmid": "1", "status": "no_conclusion", "labels": ["BACKGROUND", "DISCUSSION"], "cluster": "a", "split": "test"},
                {"pmid": "2", "status": "no_conclusion", "labels": [""], "last_words": "Further research is", "cluster": "a", "split": "test"},
                {"pmid": "3", "status": "eligible", "labels": ["CONCLUSIONS"], "cluster": "b", "split": "dev"}]
        d = R.diagnose(rows)
        self.assertEqual((d["no_conclusion"]["structured"], d["no_conclusion"]["unstructured"]), (1, 1))
        self.assertEqual(d["no_conclusion"]["labels_in_structured"][0][0], "background")
        self.assertEqual(d["largest_clusters"][0], ("a", 2, "test"))
        self.assertEqual(d["eligible_by_split"], {"dev": 1})


class DevFractionTests(unittest.TestCase):
    def test_the_share_rises_in_declared_steps_until_the_pool_is_large_enough(self):
        sizes = {f"c{i}": 1 for i in range(1000)}                      # 1000 singleton clusters
        f = spec.dev_fraction(sizes)
        got = sum(n for c, n in sizes.items() if spec.split_of(c, f) == "dev")
        self.assertGreaterEqual(got, spec.DRAFT_N)
        if f > spec.DEV_FRACTION:
            below = round(f - spec.DEV_FRACTION_STEP, 2)
            self.assertLess(sum(n for c, n in sizes.items() if spec.split_of(c, below) == "dev"), spec.DRAFT_N)
        self.assertEqual(spec.dev_fraction({f"c{i}": 1 for i in range(50)}), spec.DEV_FRACTION_MAX, "never above the cap")

    def test_every_development_cluster_stays_when_the_share_rises(self):
        names = [f"topic-{i}" for i in range(500)]
        old = {n for n in names if spec.split_of(n, 0.25) == "dev"}
        self.assertTrue(old <= {n for n in names if spec.split_of(n, 0.40) == "dev"})

    def test_prepare_records_the_share_and_the_conclusion_basis(self):
        arts = universe(200)
        recs = R.parse_efetch(("<PubmedArticleSet>" + "".join(article(p, *arts[p][:2], arts[p][3]) for p in arts) +
                               "</PubmedArticleSet>").encode())
        for r in recs:
            r["date"] = "2024-05-01"
        rows = R.prepare(recs, {p: "treatment" for p in arts})
        self.assertEqual(len({r["dev_fraction"] for r in rows}), 1)
        pools = R.pools(rows)
        self.assertIn("labelled", pools["by_conclusion_basis"])
        self.assertEqual(pools["dev_fraction"], rows[0]["dev_fraction"])


class RuleBreakdownTests(unittest.TestCase):
    def test_reasons_are_counted_without_text(self):
        rows = [{"pmid": str(i), "status": "eligible", "title": "t", "abstract": "donepezil cognition " + c, "conclusion": c}
                for i, c in enumerate(["Results were varied.", "It was effective, however results were mixed.",
                                       "No significant effect but beneficial for mood.", "Donepezil significantly improved cognition."])]
        drafts = {r["pmid"]: {"draft": {"template": "effect", "x": "donepezil", "y": "cognition", "verdict": "SUPPORTED"}} for r in rows}
        got = B.rule_breakdown(rows, drafts)
        self.assertEqual(got["well_formed_drafts"], 4)
        self.assertEqual(got["rule_no_verdict_reasons"], {"no cue": 1, "hedge with positive": 1, "conflict: positive+negative": 1})
        self.assertEqual(spec.explain_reading("Donepezil significantly improved cognition."), "label")
        wide = {"0": {"label": "NOT ENOUGH INFORMATION"}, "1": {"label": "SUPPORTED"}}
        got = B.rule_breakdown(rows, drafts, wide)["wide_verifier"]
        self.assertEqual(got["verifier_labels_all_well_formed"], {"NOT ENOUGH INFORMATION": 1, "SUPPORTED": 1})
        self.assertEqual(got["drafter_vs_verifier"]["drafter SUPPORTED / verifier NOT ENOUGH INFORMATION"], 1)


class DraftCheckTests(unittest.TestCase):
    ROW = {"title": "Donepezil and cognition", "abstract": "To assess donepezil and cognition. Donepezil significantly improved cognition.",
           "conclusion": "Donepezil significantly improved cognition."}

    def d(self, **kw):
        base = {"template": "effect", "x": "donepezil", "y": "cognition", "verdict": "SUPPORTED"}
        return dict(base, **kw)

    def test_a_good_draft_passes_every_check(self):
        self.assertEqual(B.check_draft(self.ROW, self.d()), ("rule_ok", "SUPPORTED"))

    def test_each_failure_is_named(self):
        self.assertEqual(B.check_draft(self.ROW, None)[0], "draft_unparsed")
        self.assertEqual(B.check_draft(self.ROW, self.d(verdict="NONE"))[0], "no_claim")
        self.assertEqual(B.check_draft(self.ROW, self.d(template="other"))[0], "bad_template")
        self.assertEqual(B.check_draft(self.ROW, self.d(x="aducanumab"))[0], "span_not_in_abstract")
        self.assertEqual(B.check_draft(self.ROW, self.d(verdict="REFUTED"))[0], "rule_disagrees")
        self.assertEqual(B.check_draft(dict(self.ROW, conclusion="Results were varied."), self.d())[0], "rule_no_verdict")

    def test_the_draft_reply_is_parsed_leniently_but_not_guessed(self):
        got = B.parse_draft("template: Effect\nX: \"donepezil\"\nY: cognition\nverdict: not enough information")
        self.assertEqual((got["template"], got["x"], got["verdict"]), ("effect", "donepezil", "NOT ENOUGH INFORMATION"))
        self.assertIsNone(B.parse_draft("I think it is effective."))
        self.assertIsNone(B.parse_draft("TEMPLATE: effect\nX: a\nY: b\nVERDICT: maybe")["verdict"])


class EndToEndTests(unittest.TestCase):
    def build(self, tmp, wrong_verifier=False, n=200):
        arts = universe(n)
        data, results = Path(tmp) / "data", Path(tmp) / "results"
        src = Path(tmp) / "counts.json"
        src.write_text(json.dumps(counts_file(arts, tmp)), encoding="utf-8")
        old = R.SOURCE_FILE
        R.SOURCE_FILE = src
        self.addCleanup(setattr, R, "SOURCE_FILE", old)
        common = ["--data-dir", str(data), "--results-dir", str(results)]
        quiet = contextlib.redirect_stdout(io.StringIO())
        with quiet:
            self.assertEqual(B.main(["prepare"] + common, eu=fake_eu(arts)), 0)
        model = Path(tmp) / "model.gguf"
        model.write_bytes(b"x")
        gen = scripted_generator(arts, wrong_verifier)
        with quiet:
            for split in ("dev",):
                self.assertEqual(B.main(["draft", "--split", split, "--model-path", str(model)] + common, generator=gen), 0)
                self.assertEqual(B.main(["verify", "--split", split, "--model-path", str(model)] + common, generator=gen), 0)
                self.assertEqual(B.main(["assemble", "--split", split] + common), 0)
        return arts, data, results, common, model, gen

    def test_prepare_assemble_and_the_files_that_are_tracked(self):
        with tempfile.TemporaryDirectory() as tmp:
            arts, data, results, *_ = self.build(tmp)
            pools = json.loads((results / "adkqa_pools.json").read_text(encoding="utf-8"))
            self.assertEqual(pools["records"], 200)
            dev = json.loads((results / "adkqa_build_dev.json").read_text(encoding="utf-8"))
            self.assertEqual(dev["verifier_agreement"], 1.0)
            self.assertEqual(dev["by_conclusion_basis"]["labelled"]["kept"], dev["kept"])
            self.assertEqual(dev["kept"], spec.DEV_N if dev["records_in_scope"] >= spec.DEV_N else dev["records_in_scope"])
            self.assertEqual(dev["verdict_share"]["SUPPORTED"] + dev["verdict_share"]["REFUTED"] +
                             dev["verdict_share"]["NOT ENOUGH INFORMATION"], 1.0)
            manifest = (results / "adkqa_manifest_dev.json").read_text(encoding="utf-8")
            self.assertNotIn("Alzheimer's disease?", manifest, "no question text in the tracked manifest")
            for item in json.loads(manifest)["items"]:
                self.assertEqual(set(item) & {"question", "x", "y"}, set())
            items = [json.loads(l) for l in (data / "adkqa_dev.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(i["question"].endswith("in people with Alzheimer's disease?") for i in items))

    def test_a_disagreeing_verifier_removes_every_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, results, *_ = self.build(tmp, wrong_verifier=True)
            dev = json.loads((results / "adkqa_build_dev.json").read_text(encoding="utf-8"))
            self.assertEqual(dev["kept"], 0)
            self.assertEqual(dev["verifier_agreement"], 0.0)

    def test_the_same_inputs_give_the_same_manifest_hash(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            h = [json.loads((self.build(t)[2] / "adkqa_build_dev.json").read_text(encoding="utf-8"))["manifest_hash"] for t in (a, b)]
            self.assertEqual(h[0], h[1])

    def test_a_resumed_draft_does_not_repeat_work_and_a_changed_model_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            arts, data, results, common, model, gen = self.build(tmp)
            calls = []
            with contextlib.redirect_stdout(io.StringIO()):
                B.main(["draft", "--split", "dev", "--model-path", str(model)] + common,
                       generator=lambda s, u: calls.append(1) or gen(s, u))
            self.assertEqual(calls, [])
            other = Path(tmp) / "other.gguf"
            other.write_bytes(b"y")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(B.main(["draft", "--split", "dev", "--model-path", str(other)] + common, generator=gen), 2)

    def test_the_test_split_is_refused_until_gate_one_has_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            arts, data, results, common, model, gen = self.build(tmp)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(B.main(["draft", "--split", "test", "--model-path", str(model)] + common, generator=gen), 2)
            self.assertIn("gate 1", err.getvalue())
            (results / "adkqa_gate1.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(B.main(["draft", "--split", "test", "--model-path", str(model)] + common, generator=gen), 0)


class GateTests(unittest.TestCase):
    DEV = {"survival": 0.45, "kept": 60, "verifier_agreement": 0.9, "claim_only": {"accuracy": 0.40, "majority": 0.40},
           "verdict_share": {"SUPPORTED": 0.4, "REFUTED": 0.3, "NOT ENOUGH INFORMATION": 0.3}, "manifest_hash": "h"}

    def test_pending_checks_block_the_gate_and_are_listed(self):
        g = B.evaluate_gate1(self.DEV, None, None, None)
        self.assertFalse(g["passed"])
        self.assertEqual(g["pending"], ["b0_between_constant_plus_5pp_and_80pct", "pool_median_at_least_15_and_empty_at_most_5pct",
                                        "same_seed_same_hash"])

    def test_all_checks_true_passes_and_each_threshold_bites(self):
        pools = {"median": 20, "empty_share": 0.02}
        self.assertTrue(B.evaluate_gate1(self.DEV, pools, 0.55, "h")["passed"])
        self.assertIn("same_seed_same_hash", B.evaluate_gate1(self.DEV, pools, 0.55, "other")["failed"])
        self.assertIn("b0_between_constant_plus_5pp_and_80pct", B.evaluate_gate1(self.DEV, pools, 0.42, "h")["failed"])
        self.assertIn("b0_between_constant_plus_5pp_and_80pct", B.evaluate_gate1(self.DEV, pools, 0.85, "h")["failed"])
        self.assertIn("survival_at_least_40pct", B.evaluate_gate1(dict(self.DEV, survival=0.39), pools, 0.55, "h")["failed"])
        self.assertIn("claim_only_at_most_majority_plus_5pp",
                      B.evaluate_gate1(dict(self.DEV, claim_only={"accuracy": 0.50, "majority": 0.40}), pools, 0.55, "h")["failed"])
        thin = dict(self.DEV, verdict_share={"SUPPORTED": 0.7, "REFUTED": 0.2, "NOT ENOUGH INFORMATION": 0.1})
        self.assertIn("each_verdict_at_least_25pct", B.evaluate_gate1(thin, pools, 0.8, "h")["failed"])

    def test_claim_only_classifier_finds_a_leak_and_ignores_noise(self):
        leak = [f"Is {w} effective" for w in ["alpha"] * 10 + ["beta"] * 10]
        labels = ["SUPPORTED"] * 10 + ["REFUTED"] * 10
        self.assertGreater(B.claim_only_accuracy(leak, labels)["accuracy"], 0.9)

    def test_pool_sizes_remove_the_source_and_count_empty_pools(self):
        eu = fake_eu({}, search_count=30)
        items = [{"question": "Is donepezil effective for cognition in people with Alzheimer's disease?", "date": "2024-05-01",
                  "source_pmid": "500003"}]
        got = B.pool_sizes(items, eu)
        self.assertEqual((got["median"], got["empty_share"]), (29, 0.0))
        self.assertEqual(B.pool_sizes(items, fake_eu({}, search_count=0))["empty_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
