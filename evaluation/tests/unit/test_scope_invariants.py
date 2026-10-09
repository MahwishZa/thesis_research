"""Guards against silent drift between the research design, the results and the documentation.

The repository has changed research question more than once, and each change left stale statements in several
documents that took a dedicated audit to find. These tests make the *current* state a thing that fails loudly
when contradicted: the structure of the README, the agreed objectives, the set of documents, every path, command
and section number the documents cite, the figures the README and ``docs/evaluation.md`` quote from the committed
results, and the claims about what has and has not been run.

They check documentation, which is unusual for a test suite. That is deliberate: in this repository the scope
statement is the artifact that governs every other decision, and nothing else was checking it.

History of the guards (each date is a change of what is true; older wording is in Git history):

* 2026-09-24 and 2026-10-02: repository reorganised; the research was re-scoped to an as-of evaluation on MedChangeQA
  with an Alzheimer's case study, and ``DocumentationIntegrityTests`` made dangling paths and commands fail.
* 2026-10-05: the study was realigned around an adapted RAG² baseline and an evidence-criteria verification
  extension under a supervisor requirement of +1 percentage point; the superseded Alzheimer's-specific framework left
  the active tree. The layout is checked by named *files*, not folders, because a laptop that pulls such a change
  still holds the gitignored ``corpus/data/`` and ``experiments/results/index/`` folders.
* 2026-10-06: the ``_archive`` folder was removed (Git history, commit 5e03540, keeps it).
* 2026-10-08: the code and documents of the earlier stages (recency-aware admission, evidence synthesis) left the
  active tree (Git history, commit f721bbb; their outputs are in ``experiments/medchange/results/earlier_stages/``).
  The README was reduced to six sections, ``docs/experimentation.md`` was split into ``docs/protocol.md`` and
  ``docs/evaluation.md``, ``docs/glossary.md`` and ``docs/related_work.md`` were folded into
  ``docs/methodology.md``, and the guards below were rewritten for that structure.
* 2026-10-08 (later): the README's overview, objectives and results were rewritten for a general reader (objectives as two
  bullets, no experiment status in the results), the READMEs of the subfolders were removed, and no document may point to the
  thesis paper.
* 2026-10-09: the primary evaluation was re-declared as Alzheimer's-specific (``docs/protocol.md`` §8); the as-of Cochrane
  results became secondary evidence. The README was shortened to a headline (two small tables, no experiment status), it points
  to the research paper through a placeholder (the earlier ban on mentioning the paper was temporary), and the quoted-figure
  guards were repointed to the new README and to ``docs/evaluation.md``.
"""

import ast
import json
import re
import unittest
from pathlib import Path

from experiments.medchange.scoring import detectable_range, wilson

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "README.md"
DOCS = ROOT / "docs"
RESULTS = ROOT / "experiments" / "medchange" / "results"

OBJECTIVES = (
    "To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation "
    "performance.",
    "To determine the extent to which the proposed system improves retrieval and generation performance compared "
    "with relevant baseline models and existing works.",
)
README_SECTIONS = ("Overview", "Research Objectives", "Research Results", "Repository Structure", "How to Run")
DOC_NAMES = {"data.md", "methodology.md", "protocol.md", "evaluation.md", "reproducibility.md", "log.md"}

#: Documentation that must describe the CURRENT repository exactly. ``docs/log.md`` is a dated record of what was
#: true when written and is deliberately excluded.
CURRENT_DOCS = ("README.md", "docs/data.md", "docs/methodology.md", "docs/protocol.md", "docs/evaluation.md",
                "docs/reproducibility.md")
#: Paths named in the documents that are generated or machine-local and may be absent.
GENERATED_PREFIXES = (
    "experiments/medchange/data", "experiments/adkqa/data", "experiments/medchange/results/rag2_", "experiments/medchange/results/RAG2_",
    "experiments/medchange/results/answers_", "experiments/medchange/results/label_audit_",
    "experiments/medchange/results/consistency_auto_", "experiments/adkqa/results",
)


def text(path):
    return path.read_text(encoding="utf-8")


def flat(path):
    return " ".join(text(path).split())


def headings(markdown):
    """[(level, title)] of the markdown headings, ignoring fenced code blocks."""
    out, fenced = [], False
    for line in markdown.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced:
            m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
            if m:
                out.append((len(m.group(1)), m.group(2)))
    return out


def section_numbers(path):
    """The numbered headings of a document: ``## 4. Systems`` gives "4", ``### 5.2 What R2 keeps`` gives "5.2"."""
    return set(re.findall(r"^#{2,4}\s+(\d+(?:\.\d+)*)\.?\s", text(path), flags=re.MULTILINE))


class ReadmeStructureTests(unittest.TestCase):
    """The README has exactly the agreed sections, no more."""

    def test_readme_has_a_title_and_exactly_the_agreed_sections(self):
        found = headings(text(README))
        self.assertEqual([title for level, title in found if level == 1].__len__(), 1, "exactly one title")
        self.assertEqual([title for level, title in found if level == 2], list(README_SECTIONS))
        self.assertFalse([t for level, t in found if level > 2], "no sub-headings: the README stays concise")

    def test_readme_states_exactly_the_two_objectives_verbatim(self):
        section = text(README).split("## Research Objectives", 1)[1].split("## Research Results", 1)[0]
        numbered = re.findall(r"^\d+\.\s", section, flags=re.MULTILINE)
        self.assertEqual(len(numbered), 2, "the README must list exactly two objectives")
        self.assertNotIn("|", section, "the objectives are bullet points only, with no table")
        body = " ".join(section.split())
        for objective in OBJECTIVES:
            self.assertIn(objective, body)
        remainder = body
        for number, objective in enumerate(OBJECTIVES, 1):
            remainder = remainder.replace(f"{number}. {objective}", "")
        self.assertEqual(remainder.strip(), "", "nothing but the two objectives belongs in this section")

    def test_the_results_section_presents_results_and_not_the_status_of_experiments(self):
        section = text(README).split("## Research Results", 1)[1].split("## Repository Structure", 1)[0].lower()
        for status_word in ("status", "not yet run", "not run", "planned", "implemented", "still to"):
            self.assertNotIn(status_word, section)

    def test_the_readme_points_to_the_research_paper_only_through_a_placeholder(self):
        """The research paper is not yet available: the README names it through a placeholder and invents neither an
        address nor a claim of publication."""
        body = flat(README)
        self.assertRegex(body, r"research paper: \[[^\]]*to be added\]")
        for address in ("doi.org", "arxiv.org", "researchgate", "zenodo", "scholar.google", "osf.io"):
            self.assertNotIn(address, body.lower())

    def test_readme_does_not_present_the_rag2_stand_in_as_a_reproduction(self):
        body = flat(README)
        self.assertIn("not a full reproduction", body)

    def test_readme_keeps_the_requirement_and_its_limits_visible(self):
        """The requirement is read by a pre-declared rule and a 1-point difference cannot be confirmed at this
        sample size; dropping either would turn a hedged result into an unqualified claim."""
        body = flat(README)
        for needle in ("at least 1 percentage point", "met as a point estimate, not confirmed",
                       "cannot be told apart from chance", "only gains of roughly 4 to 6 points",
                       "do not show that the extra check improves accuracy"):
            self.assertIn(needle, body)

    def test_readme_has_no_stale_headings_or_documents(self):
        body = text(README)
        for stale in ("experimentation.md", "glossary.md", "related_work.md", "Expected Contribution",
                      "Problem Statement", "Primary | Hallucination rate"):
            self.assertNotIn(stale, body)


class DocumentSetTests(unittest.TestCase):

    def test_the_docs_folder_holds_exactly_the_agreed_documents(self):
        self.assertEqual({p.name for p in DOCS.glob("*.md")}, DOC_NAMES)

    def test_the_protocol_keeps_the_decision_ledger(self):
        body = flat(DOCS / "protocol.md")
        self.assertIn("forking-path ledger", body)
        self.assertIn("Freeze the design without using the one allowed prompt revision", body)

    def test_the_evaluation_keeps_the_three_readings_of_the_requirement(self):
        body = flat(DOCS / "evaluation.md")
        for reading in ("met and confirmed", "met as a point estimate, not confirmed", "not met"):
            self.assertIn(reading, body)

    def test_the_methodology_states_what_is_held_constant(self):
        self.assertIn("held constant", text(DOCS / "methodology.md").lower())

    def test_no_current_document_cites_a_retired_document(self):
        """``experimentation.md`` was split into the protocol and the evaluation; the glossary and the related-work
        notes were folded into the methodology. ``experiment_plan.md`` (the stage-1/2 protocol) may be named only
        together with its commit, in the protocol."""
        offenders = []
        for rel in CURRENT_DOCS:
            body = text(ROOT / rel)
            for stale in ("experimentation.md", "glossary.md", "related_work.md"):
                if stale in body:
                    offenders.append(f"{rel}: {stale}")
            if "experiment_plan.md" in body and rel != "docs/protocol.md":
                offenders.append(f"{rel}: experiment_plan.md")
        self.assertEqual(offenders, [])

    def test_there_are_no_readme_files_in_the_subfolders(self):
        """The README at the root is the only one. Only the tracked folders are searched: a computer that built the
        first design also holds ignored folders (``corpus/``, ``_archive/``) with files of their own."""
        found = sorted(str(p.relative_to(ROOT)) for folder in ("docs", "src", "evaluation", "experiments")
                       for p in (ROOT / folder).rglob("README.md")
                       if "data" not in p.relative_to(ROOT).parts[:3] and "baseline" not in p.relative_to(ROOT).parts[:2])
        self.assertEqual(found, [])

    def test_section_citations_resolve(self):
        """``protocol.md §6``, ``evaluation.md §§1.3, 4`` and ``docs/methodology.md §5.2`` must name sections that
        exist; the documents cite one another by section number, so a renumbering fails here."""
        citation = re.compile(
            r"(?P<doc>protocol|evaluation|methodology|data|reproducibility)\.md`?[ \t]*§§?[ \t]*"
            r"(?P<refs>\d[\d.]*(?:[ \t]*(?:,|–|-|and)[ \t]*(?:§[ \t]*)?\d[\d.]*)*)")
        known = {name: section_numbers(DOCS / f"{name}.md")
                 for name in ("protocol", "evaluation", "methodology", "data", "reproducibility")}
        sources = [ROOT / rel for rel in CURRENT_DOCS]
        sources += [p for p in sorted(ROOT.rglob("*.py"))
                    if "__pycache__" not in p.parts and "data" not in p.relative_to(ROOT).parts[:3]
                    and p != Path(__file__).resolve()]
        missing = []
        for path in sources:
            for number, line in enumerate(text(path).splitlines(), 1):
                for m in citation.finditer(line):
                    for part in re.split(r",|\band\b", m.group("refs").replace("§", "")):
                        part = part.strip().strip(".")
                        if not part:
                            continue
                        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
                        wanted = ([str(n) for n in range(int(span.group(1)), int(span.group(2)) + 1)]
                                  if span else [part])
                        for section in wanted:
                            if section not in known[m.group("doc")]:
                                missing.append(f"{path.relative_to(ROOT)}:{number}: {m.group('doc')}.md §{section}")
        self.assertEqual(missing, [])


def _rate(cell):
    """A stored rate has a few decimals; the count it came from is exact, so rebuild it: 0.4905 of 528 is 259."""
    k = round(cell["accuracy"] * cell["n"])
    return k, cell["n"]


def _pct(k, n):
    return f"{100 * k / n:.1f}"


def _pp(x):
    return f"{100 * x:+.1f}".replace("-", "−")


def _p(x):
    return "1.0" if x >= 0.995 else f"{x:.2f}"


def _tables(markdown):
    """[(header line, [row lines])] of every markdown table."""
    out, current = [], []
    for line in markdown.splitlines() + [""]:
        if line.startswith("|"):
            current.append(line)
        elif current:
            if len(current) > 2:
                out.append((current[0], current[2:]))
            current = []
    return out


def _table(markdown, header_part):
    found = [rows for header, rows in _tables(markdown) if header_part in header]
    assert len(found) == 1, f"expected one table with {header_part!r} in its header, found {len(found)}"
    return found[0]


def _row(rows, arm):
    found = [r for r in rows if f"({arm})" in r]
    assert len(found) == 1, f"expected one row for {arm}, found {len(found)}"
    return found[0]


class DocumentsQuoteTheCommittedResultsTests(unittest.TestCase):
    """The tables and sentences of the README and of ``docs/evaluation.md`` are typed by hand; this checks them against
    the committed analyses, so a figure cannot drift from the file it claims to come from."""

    ARMS = ("B0", "B1", "R2", "R2C", "R2V", "R2V-ND")
    README_ROWS = {"B0": "No retrieval", "B1": "Standard retrieval", "R2": "Adapted RAG²", "R2V": "Proposed extra check"}

    @classmethod
    def setUpClass(cls):
        cls.confirm = json.loads(text(RESULTS / "rag2_analysis_confirm.json"))
        cls.dev = json.loads(text(RESULTS / "rag2_analysis_dev.json"))
        cls.ad = json.loads(text(RESULTS / "rag2_analysis_ad.json"))
        cls.audit = json.loads(text(RESULTS / "label_audit_confirm.json"))
        cls.report = json.loads(text(RESULTS / "report" / "report_data_confirm.json"))
        cls.report_ad = json.loads(text(RESULTS / "report_ad" / "report_data_ad.json"))
        cls.readme = text(README).replace("**", "")
        cls.readme_flat = " ".join(cls.readme.split())
        cls.evaluation = text(DOCS / "evaluation.md").replace("**", "")
        cls.evaluation_flat = " ".join(cls.evaluation.split())

    def _row(self, rows, label):
        found = [r for r in rows if r.startswith(f"| {label}")]
        self.assertEqual(len(found), 1, label)
        return found[0]

    # ---- the README: two small tables and the sentences that read them

    def test_the_readme_accuracy_table_is_quoted_exactly(self):
        rows = _table(self.readme, "System (same language model, same studies)")
        for arm, label in self.README_ROWS.items():
            cells = [f"{_pct(*_rate(sj['generation'][arm]['all']))}%" for sj in (self.confirm, self.ad)]
            row = self._row(rows, label)
            self.assertEqual([c.strip() for c in row.strip("|").split("|")[1:]], cells, label)

    def test_the_readme_difference_table_and_its_reading_are_quoted_exactly(self):
        rows = _table(self.readme, "Proposed extra check minus baseline")
        for label, sj in (("General medicine", self.confirm), ("Dementia-related", self.ad)):
            p = sj["primary"]
            expected = (f"| {label} | {_pp(p['diff_a_minus_b'])} | {_pp(p['ci95'][0])} to {_pp(p['ci95'][1])} | "
                        f"{_p(p['mcnemar_p'])} |")
            self.assertEqual(self._row(rows, label), expected)
            self.assertEqual(sj["requirement"], "met as a point estimate, not confirmed")
            self.assertGreaterEqual(100 * p["diff_a_minus_b"], 1.0, "the README says the gap is above the 1-point requirement")
            self.assertLess(p["ci95"][0], 0, "the README says the range includes negative values")
            self.assertGreater(p["ci95"][1], 0)
            self.assertFalse(p["confirmed"])
        self.assertIn("met as a point estimate, not confirmed", self.readme_flat)

    def test_the_sample_size_limits_in_the_readme_follow_from_the_formula(self):
        (a, b), (c, d) = detectable_range(self.confirm["items"]), detectable_range(self.ad["items"])
        self.assertIn(f"roughly {a} to {b} points (general medicine) or {c} to {d} points (dementia-related)",
                      self.readme_flat)

    def test_the_scope_sentences_of_the_readme_follow_from_the_data(self):
        self.assertIn(f"{self.confirm['items']} general-medicine questions and {self.ad['items']} dementia-related questions",
                      self.readme_flat)
        manifest = json.loads(text(ROOT / "experiments" / "medchange" / "manifest_ad.json"))
        self.assertEqual(manifest["items"], self.ad["items"])
        benchmark = ROOT / "experiments" / "medchange" / "data" / "benchmark.jsonl"
        if benchmark.exists():                      # local data are gitignored; the count is checked where they exist
            named = sum("alzheimer" in json.loads(line)["question"].lower() for line in text(benchmark).splitlines()
                        if line.strip() and json.loads(line)["split"] == "ad")
            self.assertIn(f"of which {named} name Alzheimer's disease", self.readme_flat)
        self.assertIn("of which 48 name Alzheimer's disease", self.readme_flat)

    def test_the_reliability_sentence_is_quoted_exactly(self):
        self.assertIn(f"reproduces {100 * self.audit['agreement']:.1f}% of them", self.readme_flat)
        for needle in (f"Agreement {100 * self.audit['agreement']:.1f}%", f"kappa {self.audit['kappa']}",
                       f"{self.audit['n_stable']} of {self.audit['n_items']} questions label-stable"):
            self.assertIn(needle, self.evaluation)

    # ---- docs/evaluation.md: the full tables

    def test_held_out_accuracies_are_quoted_exactly(self):
        gen = self.confirm["generation"]
        for arm in self.ARMS:
            k, n = _rate(gen[arm]["all"])
            lo, hi = wilson(k, n)
            full = (f"{_pct(k, n)}% ({100 * lo:.1f}–{100 * hi:.1f}) | {_pct(*_rate(gen[arm]['changed']))}% | "
                    f"{_pct(*_rate(gen[arm]['unchanged']))}%")
            self.assertIn(full, self.evaluation, f"evaluation.md, held-out {arm}")

    def test_development_accuracies_are_quoted_exactly(self):
        gen = self.dev["generation"]
        for arm in self.ARMS:
            row = " | ".join(f"{_pct(*_rate(gen[arm][kind]))}%" for kind in ("all", "changed", "unchanged"))
            self.assertIn(row, self.evaluation, f"evaluation.md, development {arm}")

    def test_the_requirement_and_its_reading_are_quoted_exactly(self):
        p = self.confirm["primary"]
        self.assertIn(f"{_pp(p['diff_a_minus_b'])} pp | {_pp(p['ci95'][0])} to {_pp(p['ci95'][1])} | {_p(p['mcnemar_p'])}",
                      self.evaluation)
        self.assertIn(self.confirm["requirement"], self.evaluation)
        d = self.dev["primary"]
        self.assertIn(f"R2V − R2 = {_pp(d['diff_a_minus_b'])} pp (95% CI {_pp(d['ci95'][0])} to {_pp(d['ci95'][1])}",
                      self.evaluation)

    def test_the_secondary_comparisons_are_quoted_exactly(self):
        for name, r in self.confirm["secondary"].items():
            verdict = "confirmed" if r["confirmed"] else "not confirmed"
            full = (f"{_pp(r['diff_a_minus_b'])} pp | {_pp(r['ci95'][0])} to {_pp(r['ci95'][1])} | "
                    f"{_p(r['mcnemar_p'])} ({_p(r['holm_p'])}) | {verdict}")
            self.assertIn(full, self.evaluation, f"evaluation.md, {name}")
        self.assertFalse(any(r["confirmed"] for r in self.confirm["secondary"].values()) or self.confirm["primary"]["confirmed"])

    def test_the_verification_and_retrieval_sentences_are_quoted_exactly(self):
        v = self.confirm["verification"]["R2V"]
        self.assertIn(f"({v['changes_fixed']} fixes, {v['changes_broke']} breaks)", self.evaluation_flat)
        r, b = self.confirm["retrieval"]["R2"], self.confirm["retrieval"]["B1"]
        self.assertIn(f"R2 admits {r['mean_admitted']:.1f} abstracts on average and none for "
                      f"{100 * r['share_without_evidence']:.1f}% of questions; {100 * r['directness_at_k']:.1f}% of its "
                      f"admitted abstracts are judged direct by the independent model, against "
                      f"{100 * b['directness_at_k']:.1f}% for B1", self.evaluation_flat)

    def test_the_existing_models_are_quoted_exactly(self):
        closed = [s for s in self.report["systems"] if "Closed-book" in s["group"]]
        self.assertEqual(len(closed), 8)
        for system in closed:
            self.assertIn(system["name"], self.evaluation_flat)
            self.assertIn(f"{100 * system['all']['accuracy']:.1f}%", self.evaluation_flat, system["name"])
        local = [100 * self.confirm["generation"][a]["all"]["accuracy"] for a in self.ARMS]
        self.assertIn(f"{min(local):.1f}–{max(local):.1f}%", self.evaluation_flat)

    def test_the_dementia_results_are_quoted_exactly(self):
        by_arm = {s["arm"]: s for s in self.report_ad["systems"]}
        for arm in ("B0", "B1", "R2", "R2C", "R2V"):
            cell = by_arm[arm]["all"]
            lo, hi = wilson(cell["correct"], cell["n"])
            self.assertIn(f"| {arm} | {_pct(cell['correct'], cell['n'])}% ({100 * lo:.1f}–{100 * hi:.1f}) |",
                          self.evaluation, arm)
        best = max(self.report_ad["constant"]["all"].values())
        self.assertIn(f"| {100 * best:.1f}% |", self.evaluation)
        p = self.ad["primary"]
        self.assertIn(f"R2V − R2 = {_pp(p['diff_a_minus_b'])} pp (95% CI {_pp(p['ci95'][0])} to {_pp(p['ci95'][1])}; "
                      f"{p['a_only_correct']} / {p['b_only_correct']} discordant, exact McNemar p = {p['mcnemar_p']:.2f})",
                      self.evaluation_flat)
        self.assertIn(self.ad["requirement"], self.evaluation)
        self.assertEqual(self.ad["requirement"], "met as a point estimate, not confirmed")


class CompletedVersusPlannedTests(unittest.TestCase):
    """What the documents call "not run" must be absent from the committed results. The checks run one way only:
    once a result exists the statement may stay stale for a while, but a result must never be claimed that does
    not exist."""

    def test_the_dementia_and_alzheimers_set_is_reported_as_not_run_until_a_result_exists(self):
        if (RESULTS / "rag2_analysis_ad.json").exists():
            self.skipTest("the dementia and Alzheimer's set has been run: update the documents")
        for rel in ("docs/protocol.md", "docs/evaluation.md", "docs/methodology.md"):
            self.assertTrue("not yet run" in flat(ROOT / rel), f"{rel} must say the set is not yet run")

    def test_the_baseline_ablations_are_reported_as_not_run_until_an_answer_exists(self):
        arms = {json.loads(line)["arm"] for line in text(RESULTS / "rag2_answers_dev.jsonl").splitlines() if line.strip()}
        if arms & {"R2-RQ", "R2-BR", "R2-NF"}:
            self.skipTest("the ablations have been run: update the documents")
        for rel in ("docs/protocol.md", "docs/methodology.md"):
            self.assertTrue("not run" in flat(ROOT / rel), f"{rel} must say the ablations were not run")

    def test_the_held_out_split_is_reported_as_run_once(self):
        self.assertTrue((RESULTS / "RAG2_FINDINGS.md").is_file())
        self.assertIn("completed once", flat(DOCS / "protocol.md"))


class SupersededScopeTests(unittest.TestCase):
    """Removed work may be remembered as history, never asserted as current."""

    def test_active_code_does_not_import_the_archive(self):
        """Nothing active may depend on the removed ``_archive`` folder."""
        hits = []
        for path in ROOT.rglob("*.py"):
            parts = path.relative_to(ROOT).parts
            if parts[0] in ("_archive", "build", "dist") or "__pycache__" in parts:
                continue
            if parts[:3] == ("experiments", "medchange", "data"):
                continue
            tree = ast.parse(text(path), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.split(".")[0] == "_archive":
                        hits.append(str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] == "_archive":
                            hits.append(str(path))
        self.assertEqual(hits, [])

    def test_the_gitignore_keeps_the_local_leftovers_out_of_git(self):
        body = text(ROOT / ".gitignore")
        for pattern in ("_archive/", "/corpus/", "experiments/results/index", "build/", "dist/"):
            self.assertIn(pattern, body)


class ActiveTreeLayoutTests(unittest.TestCase):

    #: Files (not folders: see the module docstring) that were active once and are gone: the abandoned
    #: filter-reproduction and pilot work, the Alzheimer's-specific framework, and the code of the earlier stages.
    REMOVED = (
        "experiments/shared/runners/fit_and_evaluate.py", "corpus/scripts/01_pubmed_download.py",
        "src/baseline/admission.py", "src/proposed/scorer.py", "evaluation/runner.py", "evaluation/freezing.py",
        "evaluation/questions.py", "evaluation/rag_metrics.py", "evaluation/tests/integration/test_end_to_end_runner.py",
        "experiments/medchange/stance.py", "experiments/medchange/stance_check.py",
        "experiments/medchange/synthesis.py", "experiments/medchange/analyze_stage2.py",
        "experiments/medchange/diagnostics.py", "experiments/medchange/dev_audit.py",
        "experiments/medchange/error_analysis.py", "experiments/medchange/analyze.py",
        "experiments/medchange/helpfulness.py", "experiments/medchange/findings.py",
        "experiments/medchange/pipeline.py", "evaluation/tests/unit/test_medchange_stance.py",
        "evaluation/tests/unit/test_medchange_synthesis.py", "evaluation/tests/unit/test_medchange_dev_audit.py",
        "evaluation/tests/unit/test_medchange_pipeline.py", "docs/experimentation.md", "docs/glossary.md",
        "docs/related_work.md",
    )
    ACTIVE = (
        "experiments/medchange/encoders.py", "experiments/medchange/abstracts.py", "experiments/medchange/scoring.py",
        "experiments/medchange/runner.py", "experiments/medchange/rag2.py", "experiments/medchange/rag2_run.py",
        "experiments/medchange/rag2_pipeline.py", "experiments/medchange/analyze_rag2.py",
        "src/temporal_filter/scorer.py", "src/temporal_filter/temporal.py", "src/common/evidence.py",
        "evaluation/stats.py", "docs/protocol.md", "docs/evaluation.md", "experiments/adkqa/stage0.py",
        "experiments/adkqa/spec.py", "experiments/adkqa/records.py", "experiments/adkqa/build.py",
        "experiments/adkqa/qualify.py",
    )

    def test_removed_work_is_gone_and_the_active_work_is_present(self):
        for gone in self.REMOVED:
            self.assertFalse((ROOT / gone).exists(), f"{gone} should no longer exist in the active tree")
        for kept in self.ACTIVE:
            self.assertTrue((ROOT / kept).exists(), kept)

    def test_every_active_package_is_listed_for_installation_and_nothing_else_is(self):
        """``pip install -e .`` must make exactly the active packages importable. The list in ``pyproject.toml`` is
        hand-maintained, and moving a package leaves a stale entry that no other test notices, because every test
        runs from the repository root and never needs the installed package."""
        body = text(ROOT / "pyproject.toml")
        listed = set(re.findall(r'^\s*"([\w.]+)",?\s*$', body.split("packages = [", 1)[1].split("]", 1)[0],
                                flags=re.MULTILINE))
        found = set()
        for init in ROOT.rglob("__init__.py"):
            parts = init.relative_to(ROOT).parts[:-1]
            # build output (``pip install .`` builds in the source tree: build/lib/...) is not a package
            if (not parts or parts[0] in ("_archive", "build", "dist") or parts[0].endswith(".egg-info")
                    or "__pycache__" in parts or "tests" in parts or parts[:3] == ("experiments", "medchange", "data")):
                continue
            found.add(".".join(parts))
        self.assertEqual(listed, found)

    def test_every_module_of_the_pipeline_is_in_the_readme_structure(self):
        package = ROOT / "experiments" / "medchange"
        modules = {p.name for p in package.glob("*.py") if p.name != "__init__.py"}
        block = text(README).split("## Repository Structure", 1)[1].split("```", 2)[1]
        self.assertEqual(modules - set(re.findall(r"[\w]+\.py\b", block)), set(),
                         "modules missing from the repository structure of the README")

    def test_the_repository_structure_of_the_readme_names_only_existing_files(self):
        block = text(README).split("## Repository Structure", 1)[1].split("```", 2)[1]
        names = set(re.findall(r"[\w./-]+\.(?:py|md|json|toml)\b", block))
        # only the tracked parts of the tree: a computer that built the first design also holds multi-GB ignored folders
        existing = {p.name for p in ROOT.glob("*") if p.is_file()}
        for folder in ("docs", "src", "evaluation", "experiments"):
            existing |= {p.name for p in (ROOT / folder).rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts and "data" not in p.relative_to(ROOT).parts[:3]}
        self.assertEqual(sorted(n for n in names if Path(n).name not in existing), [])


class DocumentationIntegrityTests(unittest.TestCase):
    """Every repository path and every ``python -m`` command that the current documents name must exist.
    Dangling references were the most common defect found in the audits; this makes them fail loudly."""

    PATH = re.compile(r"`((?:src|evaluation|experiments|corpus|docs|_archive)/[^`\s]*)`")
    COMMAND = re.compile(r"python -m ([A-Za-z_][\w.]*)")

    def _docs(self):
        for rel in CURRENT_DOCS:
            yield rel, text(ROOT / rel)

    def test_named_paths_exist(self):
        missing = []
        for rel, body in self._docs():
            for m in self.PATH.finditer(body):
                path = m.group(1).rstrip("/.,:;")
                if any(c in path for c in "*<>{}") or path.startswith(GENERATED_PREFIXES):
                    continue
                if not (ROOT / path).exists():
                    missing.append(f"{rel}: {path}")
        self.assertEqual(missing, [])

    def test_named_commands_exist(self):
        missing = []
        for rel, body in self._docs():
            for m in self.COMMAND.finditer(body):
                module = m.group(1)
                if module in ("unittest", "pyflakes"):
                    continue
                base = ROOT.joinpath(*module.split("."))
                if not (base.with_suffix(".py").exists() or (base / "__main__.py").exists()):
                    missing.append(f"{rel}: python -m {module}")
        self.assertEqual(missing, [])

    def test_documented_flags_exist_in_the_commands(self):
        """A flag named beside a pipeline command (``--go``, ``--judge-path``, ...) must be defined by that
        module, so that a documented command cannot silently stop working."""
        flags = {}
        for package, module in [("medchange", m) for m in (
                "rag2_pipeline", "rag2_run", "analyze_rag2", "report", "build_benchmark", "ad_benchmark", "pubmed_asof",
                "freeze_candidates", "generate_answers", "label_audit")] + [("adkqa", m) for m in ("stage0", "build", "qualify")]:
            tree = ast.parse(text(ROOT / "experiments" / package / f"{module}.py"))
            flags[(package, module)] = {a.value for node in ast.walk(tree)
                                        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                                        for a in node.args if isinstance(a, ast.Constant) and str(a.value).startswith("--")}
        missing = []
        for rel, body in self._docs():
            for line in body.splitlines():
                m = re.search(r"python -m experiments\.(medchange|adkqa)\.(\w+)(.*)", line)
                if not m or (m.group(1), m.group(2)) not in flags:
                    continue
                command = re.split(r"`|\s\|\s", m.group(3))[0]
                for flag in re.findall(r"(?<![\w-])(--[a-z][\w-]*)", command):
                    if flag not in flags[(m.group(1), m.group(2))]:
                        missing.append(f"{rel}: {m.group(2)} {flag}")
        self.assertEqual(missing, [])

    #: A repository file cited as ``dir/.../name.ext`` in a comment or docstring.
    CODE_PATH = re.compile(
        r"(?<![\w./-])((?:src|evaluation|experiments|corpus|docs|_archive)/[\w./-]*\w"
        r"\.(?:py|md|yaml|yml|json|jsonl|csv|toml|log))(?![\w-])")

    def test_paths_cited_in_code_comments_and_docstrings_exist(self):
        """Comments cite files (``docs/...``, ``src/...``) and go stale when the files move. This file is skipped
        because it names removed files on purpose."""
        missing = []
        for path in sorted(ROOT.rglob("*.py")):
            parts = path.relative_to(ROOT).parts
            if (parts[0] in ("_archive", "build", "dist") or "__pycache__" in parts
                    or parts[:3] == ("experiments", "medchange", "data") or path == Path(__file__).resolve()):
                continue
            for number, line in enumerate(text(path).splitlines(), 1):
                for m in self.CODE_PATH.finditer(line):
                    cited = m.group(1)
                    if any(c in cited for c in "*<>{}") or cited.startswith(GENERATED_PREFIXES):
                        continue
                    if not (ROOT / cited).exists():
                        missing.append(f"{'/'.join(parts)}:{number}: {cited}")
        self.assertEqual(missing, [])

    def test_current_documents_do_not_describe_removed_code_as_current(self):
        stale = ("experiments.baseline", "experiments/baseline", "next_phase_plan", "medchange.pipeline",
                 "medchange.stance", "medchange.synthesis", "medchange.analyze ", "medchange.helpfulness",
                 "medchange.findings", "analyze_stage2", "dev_audit.py", "error_analysis.py")
        offenders = []
        for rel, body in self._docs():
            for term in stale:
                if term in body:
                    offenders.append(f"{rel}: {term}")
        self.assertEqual(offenders, [])

    #: Code that moved on 2026-10-05, by its old active path. A current document that names one of these points at
    #: a path that no longer exists; a path inside another path (preceded by ``/``) is not matched.
    MOVED_CODE = re.compile(
        r"(?<![\w./-])(?:experiments/shared|experiments/results|src/baseline|src/proposed"
        r"|src/common/(?:generator|hf_generator|system)"
        r"|evaluation/(?:runner|freezing|questions|accuracy|rag_metrics|annotation)\.py"
        r"|evaluation/tests/(?:integration|corpus_scaffold)"
        r"|corpus/(?:scripts|config|metadata|reports|data|logs))")

    def test_current_documents_do_not_name_moved_code_by_its_old_path(self):
        """A computer that built the first design still holds the gitignored ``corpus/data/``, so an old path in
        a current document would even look as if it worked."""
        offenders = []
        for rel, body in self._docs():
            for number, line in enumerate(body.splitlines(), 1):
                for m in self.MOVED_CODE.finditer(line):
                    offenders.append(f"{rel}:{number}: {m.group(0)}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
