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
  extension; the superseded Alzheimer's-specific framework left the active tree. The layout is checked by named *files*, not folders, because a laptop that pulls such a change
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
* 2026-10-11: the pre-registered test on fresh questions (``docs/protocol.md`` §10, 850 questions) was in force and under way. The
  documents that omitted it were completed (``docs/data.md`` §3, ``docs/evaluation.md`` §6.9), the figures of ``docs/data.md`` were
  tied to the committed manifests, and a guard keeps the number of fresh questions equal in every place that states it.
* 2026-10-12: the documents, code and committed results state what was achieved and no longer measure the accuracy against a minimum gain
  (``StatedResultsTests``).
* 2026-10-12 (later): the code that several commands need left the command modules for libraries (``LayeringTests``), the tests were split
  to one file per module (``NamingTests``), and the committed run of the pre-registered test is checked against the registered question
  list (``CommittedFreshRunTests``).
"""

import ast
import contextlib
import hashlib
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from experiments.medchange import analyze_fresh, fresh_benchmark, pipeline_support
from experiments.medchange.files import load_jsonl
from experiments.medchange.scoring import detectable_range, wilson

ROOT = Path(__file__).resolve().parents[1]
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
    "experiments/medchange/data", "experiments/medchange/manifest_fresh", "experiments/medchange/results/rag2_", "experiments/medchange/results/RAG2_",
    "experiments/medchange/results/answers_", "experiments/medchange/results/label_audit_",
    "experiments/medchange/results/verdict_consistency_",
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

    def test_readme_states_the_result_and_keeps_its_limits_visible(self):
        """The README says what the extra check achieved and that the gaps cannot be told apart from chance at this
        sample size; dropping either would turn a hedged result into an unqualified claim."""
        body = flat(README)
        for needle in ("positive but not confirmed", "cannot be told apart from chance", "only gains of roughly 4 to 6 points",
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

    def test_the_evaluation_keeps_the_three_readings_of_a_paired_difference(self):
        body = flat(DOCS / "evaluation.md")
        for reading in ("| confirmed |", "| positive but not confirmed |", "| no evidence |"):
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
        found = sorted(str(p.relative_to(ROOT)) for folder in ("docs", "tests", "experiments")
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
        cls.report = json.loads(text(RESULTS / "report_confirm" / "report_data_confirm.json"))
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
        for label, sj in (("General medicine", self.confirm), ("Alzheimer's disease set", self.ad)):
            p = sj["primary"]
            expected = (f"| {label} | {_pp(p['diff_a_minus_b'])} | {_pp(p['ci95'][0])} to {_pp(p['ci95'][1])} | "
                        f"{_p(p['mcnemar_p'])} |")
            self.assertEqual(self._row(rows, label), expected)
            self.assertEqual(sj["reading"], "positive but not confirmed")
            self.assertGreater(p["diff_a_minus_b"], 0, "the README says the extra check scored higher")
            self.assertLess(p["ci95"][0], 0, "the README says the range includes negative values")
            self.assertGreater(p["ci95"][1], 0)
            self.assertFalse(p["confirmed"])
        self.assertIn(f"The extra check scored {100 * self.confirm['primary']['diff_a_minus_b']:.1f} points higher on general medicine and "
                      f"{100 * self.ad['primary']['diff_a_minus_b']:.1f} points higher on the Alzheimer's disease set", self.readme_flat)
        self.assertIn("they are positive but not confirmed", self.readme_flat)

    def test_the_sample_size_limits_in_the_readme_follow_from_the_formula(self):
        (a, b), (c, d) = detectable_range(self.confirm["items"]), detectable_range(self.ad["items"])
        self.assertIn(f"roughly {a} to {b} points (general medicine) or {c} to {d} points (Alzheimer's disease set)",
                      self.readme_flat)

    def test_the_scope_sentences_of_the_readme_follow_from_the_data(self):
        self.assertIn(f"{self.confirm['items']} general-medicine questions and {self.ad['items']} questions on Alzheimer's disease and related dementias",
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

    def test_the_primary_difference_and_its_reading_are_quoted_exactly(self):
        p = self.confirm["primary"]
        self.assertIn(f"{_pp(p['diff_a_minus_b'])} pp | {_pp(p['ci95'][0])} to {_pp(p['ci95'][1])} | {_p(p['mcnemar_p'])}",
                      self.evaluation)
        self.assertIn(self.confirm["reading"], self.evaluation)
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
        self.assertIn(self.ad["reading"], self.evaluation)
        self.assertEqual(self.ad["reading"], "positive but not confirmed")


class FreshResultQuotedTests(unittest.TestCase):
    """The result of the pre-registered test is typed by hand in the README, ``docs/evaluation.md`` §6.9, ``docs/protocol.md`` §10.10 and the log;
    this checks every figure against the committed analysis, so that none can drift from the file it claims to come from."""

    @classmethod
    def setUpClass(cls):
        cls.r = json.loads(text(RESULTS / "rag2_analysis_fresh.json"))
        cls.class_balance = json.loads(text(RESULTS / "class_balance.json"))["splits"]
        cls.readme = " ".join(text(README).replace("**", "").split())
        evaluation = " ".join(text(DOCS / "evaluation.md").replace("**", "").split())
        cls.evaluation = evaluation.split("### 6.9", 1)[1]
        protocol = " ".join(text(DOCS / "protocol.md").replace("**", "").replace("*", "").split())
        cls.outcome = protocol.split("### 10.10 Outcome", 1)[1]
        cls.protocol = protocol

    @staticmethod
    def f3(x):
        return f"{x:+.3f}".replace("-", "−")

    @staticmethod
    def pc(x):
        return f"{100 * x:.1f}%"

    def test_the_hypothesis_is_reported_as_not_confirmed_in_every_document(self):
        p = self.r["primary"]
        self.assertEqual(p["reading"], "no evidence")
        self.assertFalse(p["gain_of_0.02_or_more_excluded"])
        head = f"R2V − R2 = {self.f3(p['difference'])} (95% interval {self.f3(p['ci95'][0])} to {self.f3(p['ci95'][1])})"
        self.assertIn(f"Macro-F1, {head}: no evidence", self.evaluation)
        self.assertIn(f"Macro-F1, {head}, reading no evidence", self.outcome)
        self.assertIn(f"its upper end is {p['ci95'][1]:.3f}", self.evaluation)
        self.assertIn(f"the upper end is {p['ci95'][1]:.3f}", self.outcome)
        self.assertIn("H1 is not confirmed", self.outcome)
        self.assertIn("The hypothesis is not confirmed", self.evaluation)

    def test_the_readme_quotes_the_primary_and_the_accuracy_result(self):
        p, a = self.r["primary"], self.r["secondary"]["accuracy_R2V_minus_R2"]
        self.assertIn(f"the proposed check minus the baseline was {self.f3(p['difference'])} (95% range {self.f3(p['ci95'][0])} to {self.f3(p['ci95'][1])}), "
                      "which counts as no evidence of an improvement", self.readme)
        acc = self.r["accuracy"]
        self.assertIn(f"Accuracy was {self.pc(acc['R2V']['accuracy'])} with the extra check and {self.pc(acc['R2']['accuracy'])} for the baseline "
                      f"({_pp(a['diff_a_minus_b'])} points, range {_pp(a['ci95'][0])} to {_pp(a['ci95'][1])}", self.readme)
        self.assertIn("It found no evidence that the extra step improves that measure", self.readme)
        self.assertIn(str(self.r["n"]), self.readme)

    def test_the_evaluation_table_quotes_each_system(self):
        for arm, label in (("R2", "R2 (baseline)"), ("R2C", "R2C"), ("R2V", "R2V (proposed)")):
            acc, lo, hi = self.r["accuracy"][arm]["accuracy"], *self.r["accuracy"][arm]["wilson95"]
            a = self.r["arms"][arm]
            rec = " / ".join(self.pc(a["recall"][k]) for k in ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"))
            row = (f"| {label} | {self.pc(acc)} ({100 * lo:.1f}–{100 * hi:.1f}) | {a['macro_f1']:.3f} | "
                   f"{self.pc(a['predicted_share']['SUPPORTED'])} | {rec} |")
            self.assertIn(row, self.evaluation, arm)
        gold = self.r["gold_share"]
        self.assertIn("Gold verdicts of the 850 questions: SUPPORTED " + self.pc(gold["SUPPORTED"]) + ", REFUTED " + self.pc(gold["REFUTED"])
                      + ", NOT ENOUGH INFORMATION " + self.pc(gold["NOT ENOUGH INFORMATION"]), self.evaluation)

    def test_the_evaluation_quotes_the_secondary_outcomes(self):
        s = self.r["secondary"]
        m = s["macro_f1_R2V_minus_R2C"]
        self.assertIn(f"Macro-F1, R2V − R2C: {self.f3(m['difference'])} ({self.f3(m['ci95'][0])} to {self.f3(m['ci95'][1])})", self.evaluation)
        rr = s["refuted_recall_R2V_minus_R2"]
        self.assertIn(f"REFUTED recall, R2V − R2: {_pp(rr['difference'])} points ({_pp(rr['ci95'][0])} to {_pp(rr['ci95'][1])})", self.evaluation)
        a = s["accuracy_R2V_minus_R2"]
        self.assertIn(f"R2V − R2: {_pp(a['diff_a_minus_b'])} points (95% interval {_pp(a['ci95'][0])} to {_pp(a['ci95'][1])}; "
                      f"{a['a_only_correct']} / {a['b_only_correct']} discordant; exact McNemar p = {a['mcnemar_p']:.3f}), reading by §5: {a['reading']}", self.evaluation)
        c = s["accuracy_R2V_minus_R2C"]
        self.assertIn(f"R2V − R2C: {_pp(c['diff_a_minus_b'])} points ({_pp(c['ci95'][0])} to {_pp(c['ci95'][1])}; "
                      f"{c['a_only_correct']} / {c['b_only_correct']} discordant; p = {c['mcnemar_p']:.3f})", self.evaluation)

    def test_the_evaluation_quotes_what_the_verification_did(self):
        v, c = self.r["verification"]["R2V"], self.r["verification"]["R2C"]
        changed = sum(v["transitions"].values())
        self.assertEqual(round(v["changed_rate"] * v["n_with_evidence"]), changed)
        self.assertIn(f"It changed {self.pc(v['changed_rate'])} of the {v['n_with_evidence']} answers it verified ({changed}): "
                      f"{v['changes_fixed']} fixed, {v['changes_broke']} broken, {changed - v['changes_fixed'] - v['changes_broke']} stayed wrong", self.evaluation)
        self.assertIn(f"({v['transitions']['SUPPORTED -> NOT ENOUGH INFORMATION']} of {changed}) turned SUPPORTED into NOT ENOUGH INFORMATION", self.evaluation)
        self.assertIn(f"changed {self.pc(c['changed_rate'])} of the same answers: {c['changes_fixed']} fixed, {c['changes_broke']} broken", self.evaluation)

    def test_the_evaluation_quotes_the_retrieval_description(self):
        r = self.r["retrieval"]
        row = r["arms"]["R2"]
        self.assertEqual((r["arms"]["R2C"], r["arms"]["R2V"]), (row, row))                  # the three arms read the same abstracts
        mix = row["evidence_mix"]
        held_out = json.loads(text(RESULTS / "rag2_analysis_confirm.json"))["retrieval"]["R2"]
        self.assertIn(f"R2 admitted {row['mean_admitted']} abstracts per question on average and none for {self.pc(row['share_without_evidence'])} of the questions "
                      f"(held-out split: {held_out['mean_admitted']} and {self.pc(held_out['share_without_evidence'])})", self.evaluation)
        self.assertIn(f"{self.pc(mix['SR/MA'])} are systematic reviews or meta-analyses, {self.pc(mix['RCT'])} trials and {self.pc(mix['other'])} other studies", self.evaluation)
        self.assertIn(f"their mean age at the review date is {row['mean_age_years']} years", self.evaluation)
        self.assertIn(f"{r['single_version']} of the {self.r['n']} questions come from reviews with a single version", self.evaluation)
        self.assertIn("10. Description of the retrieval (2026-10-12)", self.protocol)
        self.assertIn("## Retrieval (descriptive)", text(RESULTS / "RAG2_FINDINGS_FRESH.md"))

    def test_the_descriptive_comparison_and_the_label_audit_are_quoted(self):
        d = self.r["alzheimers_dementia_descriptive"]
        m = d["macro_f1_R2V_minus_R2"]
        self.assertFalse(d["sign_agrees_with_fresh"])
        self.assertIn(f"on its {d['n']} questions: {self.f3(m['difference'])} ({self.f3(m['ci95'][0])} to {self.f3(m['ci95'][1])}); its sign differs", self.evaluation)
        au = self.r["label_audit"]
        self.assertIn(f"In the sample of {au['n_items']}, agreement with the independent model {self.pc(au['agreement'])}, kappa {au['kappa']}", self.evaluation)

    def test_the_reading_compares_the_refuted_recall_with_the_earlier_sets(self):
        recall = lambda split, arm: self.class_balance[split]["arms"][arm]["recall"]["REFUTED"]
        fresh = lambda arm: self.r["arms"][arm]["recall"]["REFUTED"]
        for split, label in (("confirm", "held-out split"), ("ad", "Alzheimer's disease set")):
            gain = recall(split, "R2V") - recall(split, "R2")
            self.assertIn(f"{100 * gain:.1f} points on the {label} ({self.pc(recall(split, 'R2'))} to {self.pc(recall(split, 'R2V'))})", self.evaluation)
        self.assertIn(f"by {100 * (fresh('R2V') - fresh('R2')):.1f} points here ({self.pc(fresh('R2'))} to {self.pc(fresh('R2V'))})", self.evaluation)
        for arm in ("R2", "R2V"):
            self.assertIn(self.pc(self.r["arms"][arm]["recall"]["NOT ENOUGH INFORMATION"]), self.evaluation)
        r2c = self.r["arms"]["R2C"]
        self.assertIn(f"(macro-F1 {r2c['macro_f1']:.3f}, accuracy {self.pc(self.r['accuracy']['R2C']['accuracy'])})", self.evaluation)
        self.assertIn("81%", self.evaluation)                      # the registered power at 0.04 and 0.045, checked against the formula in FreshTestDocumentsTests
        self.assertIn("89%", self.evaluation)

    def test_the_status_row_of_the_protocol_and_the_methodology_report_the_result(self):
        p = self.r["primary"]
        self.assertIn(f"macro-F1, R2V − R2 = {self.f3(p['difference'])}, 95% interval {self.f3(p['ci95'][0])} to {self.f3(p['ci95'][1])}: no evidence", self.protocol)
        self.assertIn("It found no evidence that it does", " ".join(text(DOCS / "methodology.md").split()))
        self.assertIn("## Phase 77", text(DOCS / "log.md"))

    def test_the_findings_file_is_the_rendering_of_the_committed_analysis(self):
        md = text(RESULTS / "RAG2_FINDINGS_FRESH.md")
        p = self.r["primary"]
        self.assertIn(f"Macro-F1, R2V − R2 = **{p['difference']:+.3f}** (95% interval {p['ci95'][0]:+.3f} to {p['ci95'][1]:+.3f}; {self.r['iterations']} bootstrap resamples).", md)
        self.assertIn(f"Reading: **{p['reading']}**.", md)
        self.assertNotIn("not met", md)


class AnswersAndLimitationsTests(unittest.TestCase):
    """Sections 7 (answers to the research questions) and 8 (limitations) of ``docs/evaluation.md`` are typed by hand; every figure in them is checked
    against the committed file it comes from (2026-10-12)."""

    @classmethod
    def setUpClass(cls):
        body = " ".join(text(DOCS / "evaluation.md").replace("**", "").split())
        cls.s7 = body.split("## 7. Answers to the research questions", 1)[1].split("## 8. Limitations and threats to validity", 1)[0]
        cls.s8 = body.split("## 8. Limitations and threats to validity", 1)[1]
        load = lambda name: json.loads(text(RESULTS / name))
        cls.confirm, cls.ad, cls.fresh = load("rag2_analysis_confirm.json"), load("rag2_analysis_ad.json"), load("rag2_analysis_fresh.json")
        cls.report = load("report_confirm/report_data_confirm.json")

    @staticmethod
    def acc(rep, arm):
        return _pct(*_rate(rep["generation"][arm]["all"]))

    @staticmethod
    def ci(r):
        return f"{_pp(r['ci95'][0])} to {_pp(r['ci95'][1])}"

    def test_the_sections_exist_and_answer_every_question_of_the_protocol(self):
        for needle in ("Primary (`protocol.md` §1)", "(a)", "(b)", "(c)", "(d)", "Pre-registered", "Objective 1", "Objective 2"):
            self.assertIn(needle, self.s7)
        self.assertIn("(d) How do the evidence sets that R2 admits differ from those of B1?", self.s7)
        protocol = flat(DOCS / "protocol.md")
        for question in ("(a) Does the adapted RAG² baseline beat standard retrieval", "(b) How much of any gain comes from stating the evidence criteria alone",
                         "(c) Does the currency criterion", "(d) How do the evidence sets that R2 admits differ from B1's"):
            self.assertIn(question, protocol)

    def test_the_three_primary_answers_are_those_of_the_committed_analyses(self):
        c, a, f = self.confirm["primary"], self.ad["primary"], self.fresh["primary"]
        self.assertIn(f"{self.acc(self.confirm, 'R2V')}% against {self.acc(self.confirm, 'R2')}% (528 questions): {_pp(c['diff_a_minus_b'])} points, 95% interval {self.ci(c)}, "
                      f"exact McNemar p = {c['mcnemar_p']:.2f} (§6.2)", self.s7)
        self.assertIn(f"{self.acc(self.ad, 'R2V')}% against {self.acc(self.ad, 'R2')}%: {_pp(a['diff_a_minus_b'])} points, {self.ci(a)}, p = {a['mcnemar_p']:.2f} (§6.6)", self.s7)
        m = lambda arm: f"{self.fresh['arms'][arm]['macro_f1']:.3f}"
        f3 = lambda x: f"{x:+.3f}".replace("-", "−")
        self.assertIn(f"{m('R2V')} against {m('R2')}: {f3(f['difference'])}, 95% interval {f3(f['ci95'][0])} to {f3(f['ci95'][1])} (§6.9)", self.s7)
        self.assertEqual((self.confirm["reading"], self.ad["reading"], f["reading"]), ("positive but not confirmed", "positive but not confirmed", "no evidence"))

    def test_the_secondary_answers_quote_the_comparisons(self):
        cs, as_ = self.confirm["secondary"], self.ad["secondary"]
        diff = lambda r: f"{_pp(r['diff_a_minus_b'])} ({self.ci(r)})"
        self.assertIn(f"Held-out: R2 {self.acc(self.confirm, 'R2')}%, B1 {self.acc(self.confirm, 'B1')}%, B0 {self.acc(self.confirm, 'B0')}%; R2 − B1 = {_pp(cs['R2 vs B1']['diff_a_minus_b'])} points ({self.ci(cs['R2 vs B1'])}). "
                      f"Alzheimer's disease set: R2 {self.acc(self.ad, 'R2')}%, B1 {self.acc(self.ad, 'B1')}%, B0 {self.acc(self.ad, 'B0')}%; R2 − B1 = {diff(as_['R2 vs B1'])}", self.s7)
        self.assertIn(f"Held-out: R2C − R2 = {diff(cs['R2C vs R2'])}, R2V − R2C = {diff(cs['R2V vs R2C'])}. Alzheimer's disease set: {diff(as_['R2C vs R2'])} and {diff(as_['R2V vs R2C'])}.", self.s7)
        s = self.fresh["secondary"]["macro_f1_R2V_minus_R2C"]
        f3 = lambda x: f"{x:+.3f}".replace("-", "−")
        self.assertIn(f"Fresh macro-F1: R2V − R2C = {f3(s['difference'])} ({f3(s['ci95'][0])} to {f3(s['ci95'][1])})", self.s7)
        self.assertIn(f"Held-out: {_pp(cs['R2V vs R2V-ND']['diff_a_minus_b'])} points ({self.ci(cs['R2V vs R2V-ND'])}); not run on the other sets", self.s7)
        self.assertFalse(any(v["confirmed"] for v in list(cs.values()) + list(as_.values())))

    def test_the_retrieval_answer_and_the_objectives_quote_the_files(self):
        r, b = self.confirm["retrieval"]["R2"], self.confirm["retrieval"]["B1"]
        pc = lambda x: f"{100 * x:.1f}%"
        self.assertIn(f"R2 admits {r['mean_admitted']} abstracts per question against {b['mean_admitted']} and none for {pc(r['share_without_evidence'])} of the questions; "
                      f"systematic reviews or meta-analyses are {pc(r['evidence_mix']['SR/MA'])} of its abstracts against {pc(b['evidence_mix']['SR/MA'])}, trials "
                      f"{pc(r['evidence_mix']['RCT'])} against {pc(b['evidence_mix']['RCT'])}; {pc(r['directness_at_k'])} are judged direct by the independent model against {pc(b['directness_at_k'])}", self.s7)
        p = {k: self.confirm["primary"] if k == "confirm" else self.ad["primary"] for k in ("confirm", "ad")}
        fresh_difference = f"{self.fresh['primary']['difference']:+.3f}".replace("-", "−")
        self.assertIn(f"{_pp(p['confirm']['diff_a_minus_b'])} and {_pp(p['ad']['diff_a_minus_b'])} points and the macro-F1 difference on new questions was {fresh_difference}", self.s7)
        self.assertIn(f"gains of {100 * p['confirm']['ci95'][1]:.1f}, {100 * p['ad']['ci95'][1]:.1f} and {self.fresh['primary']['ci95'][1]:.3f} (macro-F1)", self.s7)
        released = {x["name"]: x["all"]["accuracy"] for x in self.report["systems"] if x["group"].startswith("Closed-book")}
        self.assertEqual((min(released, key=released.get), max(released.values())), ("PMC-LLaMA", released["Qwen2.5-7B"]))
        self.assertIn(f"from {pc(min(released.values()))} (PMC-LLaMA) to {pc(max(released.values()))} (Qwen2.5-7B and GPT-4o-mini)", self.s7)
        local = [self.confirm["generation"][a]["all"]["accuracy"] for a in ("B0", "B1", "R2", "R2C", "R2V", "R2V-ND")]
        self.assertIn(f"the local systems ({pc(min(local))} to {pc(max(local))})", self.s7)

    def test_every_statement_of_the_label_reproducibility_in_the_current_documents_is_the_audit_value(self):
        audit = {"held-out": json.loads(text(RESULTS / "label_audit_confirm.json")), "development": json.loads(text(RESULTS / "label_audit_dev.json"))}
        seen = 0
        for rel in CURRENT_DOCS:
            body = " ".join(text(ROOT / rel).replace("**", "").split())
            for m in re.finditer(r"(\d+\.\d)% of the (held-out|development) (?:gold )?labels", body):
                seen += 1
                self.assertEqual(m.group(1), f"{100 * audit[m.group(2)]['agreement']:.1f}", f"{rel}: {m.group(0)}")
        self.assertGreaterEqual(seen, 3)

    def test_the_limitations_quote_the_audits_the_sample_sizes_and_the_sets(self):
        audit = lambda split: json.loads(text(RESULTS / f"label_audit_{split}.json"))
        pc = lambda x: f"{100 * x:.1f}%"
        self.assertIn(f"{pc(audit('confirm')['agreement'])} of the held-out labels (kappa {audit('confirm')['kappa']}), {pc(audit('dev')['agreement'])} of the development labels and "
                      f"{pc(audit('fresh')['agreement'])} of a sample of {audit('fresh')['n_items']} fresh labels (kappa {audit('fresh')['kappa']})", self.s8)
        audited = (RESULTS / "label_audit_ad.json").exists()
        self.assertEqual("have not been audited" in self.s8, not audited, "§8 must say whether the labels of the Alzheimer's disease set have been audited")
        self.assertEqual("the label audit of the Alzheimer's disease set," in self.s8, not audited)
        lo, hi = detectable_range(528)
        lo2, hi2 = detectable_range(208)
        self.assertIn(f"With 528 questions only gains of roughly {lo} to {hi} points can be confirmed, with 208 only {lo2} to {hi2}", self.s8)
        manifests = {n: json.loads(text(PACKAGE / f"manifest{n}.json")) for n in ("", "_ad", "_fresh")}
        ad, fresh = manifests["_ad"], manifests["_fresh"]
        self.assertIn(f"{ad['items']} questions come from {ad['reviews']} reviews", self.s8)
        self.assertIn(f"Of the {fresh['items']} fresh questions, {fresh['single_version']} come from reviews with a single version, and so do {ad['single_version']} of the {ad['items']} of the Alzheimer's disease set", self.s8)
        counts = manifests[""]["counts"]["ad_related"]
        self.assertIn(f"The main benchmark has {sum(counts.values())} Alzheimer's-related questions", self.s8)
        changed = self.confirm["generation"]["B0"]["changed"]["n"]
        self.assertIn(f"({changed} of the {self.confirm['items']} held-out questions)", self.s8)
        supply = json.loads(text(RESULTS / "ad_cochrane_supply.json"))
        self.assertIn(f"{supply['usable']} usable new reviews were found, 30 were needed", self.s8)
        self.assertEqual(json.loads(text(RESULTS / "ad_subgroup_alzheimer.json"))["n"], 48)
        self.assertIn("48 questions name Alzheimer's disease", self.s8)


class CompletedVersusPlannedTests(unittest.TestCase):
    """What the documents call "not run" must be absent from the committed results. The checks run one way only:
    once a result exists the statement may stay stale for a while, but a result must never be claimed that does
    not exist."""

    def test_the_alzheimers_disease_set_is_reported_as_run_once_because_its_result_exists(self):
        """Its analysis is committed, so the documents must report it as run, and none may still call it "not yet run"."""
        self.assertTrue((RESULTS / "rag2_analysis_ad.json").exists())
        self.assertIn("completed once", flat(DOCS / "protocol.md"))
        self.assertIn("run once", flat(DOCS / "evaluation.md"))
        for rel in CURRENT_DOCS:
            self.assertNotIn("not yet run", flat(ROOT / rel), rel)

    def test_the_fresh_test_is_reported_as_completed_because_its_analysis_is_committed(self):
        """Its blinded analysis is committed (2026-10-12), so the documents must report it as run and none may still say that no result exists
        (the figures themselves are checked by ``FreshResultQuotedTests``)."""
        self.assertTrue((RESULTS / "rag2_analysis_fresh.json").exists())
        self.assertTrue((RESULTS / "RAG2_FINDINGS_FRESH.md").exists())
        self.assertIn("### 10.10 Outcome", text(DOCS / "protocol.md"))
        self.assertIn("completed once", flat(DOCS / "protocol.md"))
        self.assertNotIn("None yet", flat(DOCS / "evaluation.md").split("### 6.9", 1)[1])

    def test_the_baseline_ablations_are_reported_as_not_run_until_an_answer_exists(self):
        arms = {json.loads(line)["arm"] for line in text(RESULTS / "rag2_answers_dev.jsonl").splitlines() if line.strip()}
        if arms & {"R2-RQ", "R2-BR", "R2-NF"}:
            self.skipTest("the ablations have been run: update the documents")
        for rel in ("docs/protocol.md", "docs/methodology.md"):
            self.assertTrue("not run" in flat(ROOT / rel), f"{rel} must say the ablations were not run")

    def test_the_held_out_split_is_reported_as_run_once(self):
        self.assertTrue((RESULTS / "RAG2_FINDINGS_CONFIRM.md").is_file())
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


class DataDocumentQuotesTheManifestsTests(unittest.TestCase):
    """``docs/data.md`` states counts by hand; they must be those of the committed manifests (one of them, 397 of 504, had a wrong denominator)."""

    @classmethod
    def setUpClass(cls):
        base = ROOT / "experiments" / "medchange"
        cls.main = json.loads(text(base / "manifest.json"))
        cls.ad = json.loads(text(base / "manifest_ad.json"))
        cls.fresh = json.loads(text(base / "manifest_fresh.json"))
        cls.supply = json.loads(text(RESULTS / "fresh_supply.json"))
        cls.data = flat(DOCS / "data.md")

    def test_the_main_benchmark(self):
        c = self.main["counts"]
        usable = c["items"] - c["likely_label_noise_excluded"]
        self.assertIn(f"{usable} usable Cochrane questions", self.data)
        self.assertIn(f"{c['usable_changed']} usable changed questions", self.data)
        self.assertIn(f"samples {usable - c['usable_changed']} unchanged controls", self.data)
        self.assertIn(f"{c['likely_label_noise_excluded']} changed items", self.data)
        nei = sum(n for k, n in c["change_types"].items() if "NOT ENOUGH INFORMATION" in k)
        self.assertIn(f"{nei} of the {c['usable_changed']} usable changes ({round(100 * nei / c['usable_changed'])}%) involve NOT ENOUGH INFORMATION", self.data)
        self.assertEqual({k.split("/")[1] for k in c["by_kind_split"]}, {"dev", "confirm"})
        self.assertEqual(sum(c["by_kind_split"].values()), c["items"])

    def test_the_alzheimers_disease_set(self):
        a = self.ad
        self.assertIn(f"{a['items']} questions from {a['reviews']} reviews", self.data)
        for label, key in (("NOT ENOUGH INFORMATION", "NOT ENOUGH INFORMATION"), ("REFUTED", "REFUTED"), ("SUPPORTED", "SUPPORTED")):
            self.assertIn(f"{label} {a['labels'][key]} ({round(100 * a['labels'][key] / a['items'])}%)", self.data)
        self.assertIn(f"{a['single_version']} come from reviews with a single version", self.data)
        self.assertEqual(a["kinds"], {"unchanged": a["items"]})

    def test_the_fresh_questions(self):
        f, sup = self.fresh, self.supply["by_earliest_date"]["2010-01-01"]
        self.assertIn(f"{f['items']} questions from {f['reviews']} reviews, all of unchanged verdict", self.data)
        self.assertIn(f"{f['single_version']} come from reviews with a single version", self.data)
        for label in ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"):
            self.assertIn(f"{label} {f['labels'][label]} ({100 * f['labels'][label] / f['items']:.1f}%)", self.data)
        lo, hi = f["selection"]["dates"]
        self.assertIn(f"dated {lo} to {hi}", self.data)
        self.assertIn(f"{sup['questions']:,} questions from {sup['reviews']:,} reviews", self.data)
        self.assertEqual(f["selection"]["qualifying_questions"], sup["questions"])
        self.assertEqual(f["selection"]["selected"], f["items"])
        self.assertIn(f"{self.supply['fresh_questions_all_dates']:,} of the 16,501 rows", self.data)
        self.assertEqual(self.supply["medrevqa_rows"], 16501)
        self.assertIn(f"only {sup['dementia_in_review_text']['questions']} name Alzheimer's disease or dementia anywhere", self.data)
        self.assertIn(f"{sup['neurodegenerative_wording']['questions']} name another neurodegenerative disease in the question", self.data)


class FreshTestDocumentsTests(unittest.TestCase):
    """The pre-registered test is described in several documents; the number of questions and the date it came into force must agree everywhere."""

    N = json.loads(text(ROOT / "experiments" / "medchange" / "manifest_fresh.json"))["items"]

    def test_every_document_that_describes_the_test_states_the_registered_number_of_questions(self):
        for rel in ("README.md", "docs/data.md", "docs/methodology.md", "docs/protocol.md", "docs/evaluation.md", "docs/reproducibility.md"):
            self.assertIn(str(self.N), flat(ROOT / rel), rel)

    def test_the_protocol_is_in_force_consistently(self):
        body = text(DOCS / "protocol.md")
        heading = re.search(r"^## 10\. Pre-registration of a confirmatory test on fresh questions \((.*)\)$", body, flags=re.MULTILINE)
        status = re.search(r"\*\*Status\.\*\* This section is IN FORCE since (\d{4}-\d{2}-\d{2})", body)
        self.assertIsNotNone(heading, "the heading of §10")
        self.assertIsNotNone(status, "the status paragraph of §10")
        self.assertEqual(heading.group(1), f"in force since {status.group(1)}")
        self.assertIn(f"in force since {status.group(1)}", flat(DOCS / "protocol.md"))
        self.assertNotIn("DRAFT", body.split("## 10.", 1)[1].split("### 10.1", 1)[0])
        self.assertIn(f"N = {self.N}", flat(DOCS / "protocol.md"))

    def test_the_status_table_and_the_ledger_record_the_test(self):
        body = flat(DOCS / "protocol.md")
        for needle in ("Pre-registered test on fresh questions", "Reduce N from 1,500 to 850", "mark §10 in force"):
            self.assertIn(needle, body)

    def test_the_power_figures_follow_from_the_standard_error(self):
        """0.41 / sqrt(N) is the standard error of the registered power statement; the quoted percentages must follow from it."""
        from math import erf, sqrt
        phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
        se = 0.41 / sqrt(self.N)
        body = flat(DOCS / "protocol.md")
        for gain, quoted in ((0.04, "81%"), (0.03, "57%"), (0.02, "30%"), (0.045, "89%")):
            self.assertEqual(f"{round(100 * phi(gain / se - 1.96))}%", quoted)
            self.assertIn(quoted, body)
        self.assertIn(f"about {2.8 * se:.3f}", body)


class CompositionOfTheAlzheimersDiseaseSetTests(unittest.TestCase):
    """The set is described by the wording of its questions: 48 name Alzheimer's disease, 156 dementia, 19 neither, and 15 name both, so the
    three numbers add up to 223, not 208. Every statement of the composition says so (2026-10-12)."""

    def test_the_numbers_come_from_the_committed_files(self):
        sub = json.loads(text(RESULTS / "ad_subgroup_alzheimer.json"))
        self.assertEqual((sub["n"], sub["dementia_set_items"]), (48, 208))
        self.assertEqual(48 + 156 - 15 + 19, 208)
        self.assertIn("48 name Alzheimer's disease and 156 dementia (15 name both) and 19 name neither", flat(DOCS / "protocol.md").replace("\n", " "))

    def test_every_statement_of_the_composition_gives_the_overlap(self):
        offenders = []
        for rel in CURRENT_DOCS:
            body = " ".join(text(ROOT / rel).split())
            for m in re.finditer(r"156 (?:name )?dementia", body):
                if not body[m.end():m.end() + 20].startswith(" (15 name both)"):
                    offenders.append(f"{rel}: {body[max(0, m.start() - 40):m.end() + 40]}")
        self.assertEqual(offenders, [])


class TerminologyTests(unittest.TestCase):
    """The researcher asked for the full form, or "Alzheimer's disease" for short, and not for an abbreviation of "Alzheimer's disease and related
    dementias"; the 208-question set is called the Alzheimer's disease set."""

    def test_the_abbreviation_is_not_used_in_any_document_or_source_file(self):
        offenders = []
        files = [ROOT / rel for rel in CURRENT_DOCS] + [DOCS / "log.md"]
        files += [p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts and "data" not in p.relative_to(ROOT).parts[:3]]
        for path in files:
            if re.search(r"\bADRD\b", text(path)) and path != Path(__file__).resolve():
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_the_set_is_not_called_a_dementia_set_in_the_current_descriptive_text(self):
        """History (the ledger, the amendments, the log) keeps the words it was written in; the README, the data description and the
        evaluation results do not call the 208 questions "the dementia set" or "the dementia run"."""
        offenders = []
        for rel in ("README.md", "docs/data.md", "docs/evaluation.md", "docs/methodology.md", "docs/reproducibility.md"):
            for number, line in enumerate(text(ROOT / rel).splitlines(), 1):
                if re.search(r"dementia[- ]set|dementia run|dementia split", line, flags=re.IGNORECASE) and "(`ad`)" not in line:
                    offenders.append(f"{rel}:{number}")
        self.assertEqual(offenders, [])


class StatedResultsTests(unittest.TestCase):
    """The researcher asked (2026-10-12) that the repository state what was achieved and no longer measure it against a minimum gain
    of one point. Differences are read as confirmed, positive but not confirmed, or no evidence (``docs/evaluation.md`` §5). The
    dated record of the change (``docs/protocol.md`` §8 and §10.9), the log and the results of the earlier stages keep their words."""

    OLD_WORDING = re.compile(
        r"1[- ]point requirement|requirement of \+1|met as a point estimate|at least 1(?:\.0)? (?:percentage point|pp)|"
        r"\+1(?:\.0)? (?:pp|point)|REQUIREMENT_PP|requirement_reading|\"requirement\"|\bthe requirement\b", re.IGNORECASE)

    def _protocol_without_the_dated_record(self):
        body = text(DOCS / "protocol.md")
        start, stop = body.index("**2026-10-12, how the accuracy differences are read.**"), body.index("## 9. Earlier stages")
        body = body[:start] + body[stop:]
        item = body.index("7. *Wording of the readings.*")
        return body[:item]

    def test_the_current_documents_do_not_measure_the_accuracy_against_a_minimum_gain(self):
        offenders = []
        for rel in CURRENT_DOCS:
            body = self._protocol_without_the_dated_record() if rel == "docs/protocol.md" else text(ROOT / rel)
            offenders += [f"{rel}: {m.group(0)}" for m in self.OLD_WORDING.finditer(re.sub(r"\s+", " ", body))]      # a phrase may be split over two lines
        self.assertEqual(offenders, [])

    def test_the_code_and_the_committed_results_do_not_measure_the_accuracy_against_a_minimum_gain(self):
        files = list((ROOT / "experiments" / "medchange").glob("*.py")) + list((ROOT / "experiments" / "medchange" / "stage1").glob("*.py"))
        files += [p for p in RESULTS.glob("*.md")] + [p for p in RESULTS.glob("*.json") if "answers" not in p.name and "rationales" not in p.name]
        files += list((RESULTS / "report_confirm").glob("*.md")) + list((RESULTS / "report_ad").glob("*.md"))
        offenders = [str(p.relative_to(ROOT)) for p in files if self.OLD_WORDING.search(re.sub(r"\s+", " ", text(p)))]
        self.assertEqual(offenders, [])

    def test_the_change_is_recorded_once_in_the_protocol_and_in_the_log(self):
        body = text(DOCS / "protocol.md")
        self.assertEqual(body.count("**2026-10-12, how the accuracy differences are read.**"), 1)
        self.assertIn("| 2026-10-12 | Read accuracy differences by the confirmation rule alone", body)
        self.assertIn("7. *Wording of the readings.*", body)
        self.assertIn("## Phase 75", text(DOCS / "log.md"))


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
        # the tests moved to ``tests/`` on 2026-10-12 (the bytecode folders they leave behind on a computer that ran them are not files of the tree)
        "evaluation/tests/__init__.py", "evaluation/tests/check_hermetic.py", "evaluation/tests/unit/__init__.py",
        "evaluation/tests/unit/test_scope_invariants.py", "evaluation/tests/unit/test_medchange_rag2.py",
        "experiments/medchange/subgroup_ad.py", "experiments/medchange/cochrane_ad_supply.py",
        "experiments/medchange/consistency_auto.py",
        # folded into the pipeline package on 2026-10-12 (``src/`` and ``evaluation/`` no longer exist), renamed or moved
        "src/__init__.py", "src/common/__init__.py", "src/common/evidence.py", "src/temporal_filter/__init__.py",
        "src/temporal_filter/scorer.py", "src/temporal_filter/temporal.py", "evaluation/__init__.py", "evaluation/stats.py",
        "experiments/medchange/runner.py", "experiments/medchange/results/RAG2_FINDINGS.md",
        "tests/test_runner.py", "tests/test_temporal_filter.py",
        # split to one test file per module on 2026-10-12 (the shared code left the command modules the same day)
        "tests/test_audits.py", "tests/test_fresh_pipeline.py",
    )
    ACTIVE = (
        "experiments/medchange/encoders.py", "experiments/medchange/abstracts.py", "experiments/medchange/scoring.py",
        "experiments/medchange/pipeline_support.py", "experiments/medchange/rag2.py", "experiments/medchange/rag2_run.py",
        "experiments/medchange/rag2_pipeline.py", "experiments/medchange/analyze_rag2.py",
        "experiments/medchange/stage1/scorer.py", "experiments/medchange/stage1/temporal.py", "experiments/medchange/stage1/evidence.py",
        "experiments/medchange/stats.py", "tests/check_hermetic.py", "docs/protocol.md", "docs/evaluation.md",
        "experiments/medchange/analysis.py", "experiments/medchange/eutils.py", "experiments/medchange/files.py",
        "experiments/medchange/generator.py", "experiments/medchange/selection.py", "experiments/medchange/released_answers.py",
        "tests/fixtures.py",
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
        for folder in ("docs", "tests", "experiments"):
            existing |= {p.name for p in (ROOT / folder).rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts and "data" not in p.relative_to(ROOT).parts[:3]}
        self.assertEqual(sorted(n for n in names if Path(n).name not in existing), [])


class NamingTests(unittest.TestCase):
    """Names say what a file is (2026-10-12): the tests live in ``tests/``, one file per module (``test_<module>.py``) plus the guards; the modules about the
    Alzheimer's disease set start with ``ad_``; the committed reports are in one folder per split; no module or output keeps the vestigial
    suffix ``_auto`` (every check of this study is automatic)."""

    PACKAGE = ROOT / "experiments" / "medchange"
    #: The one test file that is not about a module: it ties the code, the documents and the committed results together.
    GUARDS = {"repository_guards"}

    def _modules(self):
        """The modules of the pipeline package and its subpackages (``stage1``)."""
        modules = {p.stem for p in self.PACKAGE.glob("*.py") if p.name != "__init__.py"}
        return modules | {p.name for p in self.PACKAGE.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}

    def test_every_test_file_is_in_tests_and_named_after_a_module(self):
        stray = sorted(str(p.relative_to(ROOT)) for folder in ("experiments", "docs")
                       for p in (ROOT / folder).rglob("test_*.py") if "__pycache__" not in p.parts)
        self.assertEqual(stray, [], "tests belong in tests/")
        allowed = self._modules() | {"check_hermetic"} | self.GUARDS
        misnamed = sorted(p.name for p in (ROOT / "tests").glob("test_*.py") if p.stem[len("test_"):] not in allowed)
        self.assertEqual(misnamed, [])

    def test_every_module_has_its_test_file(self):
        tested = {p.stem[len("test_"):] for p in (ROOT / "tests").glob("test_*.py")}
        self.assertEqual(sorted(self._modules() - tested), [], "modules without a test file")

    def test_modules_about_the_alzheimers_disease_set_start_with_ad(self):
        misplaced = sorted(m for m in self._modules() if re.search(r"(^|_)ad(_|$)", m) and not m.startswith("ad_"))
        self.assertEqual(misplaced, [])

    def test_no_module_or_committed_output_keeps_the_auto_suffix(self):
        names = [m for m in self._modules() if m.endswith("_auto") or "_auto_" in m]
        names += [p.name for p in RESULTS.iterdir() if "_auto" in p.name]
        self.assertEqual(names, [])

    def test_the_folding_of_src_and_evaluation_is_recorded(self):
        """``src/`` and ``evaluation/`` were folded into the pipeline package on 2026-10-12 (Phase 78 of the log); the pipeline's own record of the change is in the protocol."""
        self.assertIn("## Phase 78", text(DOCS / "log.md"))
        self.assertIn("Phase 78", text(DOCS / "protocol.md"))
        self.assertTrue((self.PACKAGE / "stage1" / "__init__.py").is_file())
        self.assertTrue((self.PACKAGE / "stats.py").is_file())

    def test_the_committed_reports_are_one_folder_per_split(self):
        folders = sorted(p.name for p in RESULTS.iterdir() if p.is_dir() and p.name.startswith("report"))
        self.assertEqual(folders, ["report_ad", "report_confirm"])
        for split in ("ad", "confirm"):
            self.assertTrue((RESULTS / f"report_{split}" / "REPORT.md").is_file())


PACKAGE = ROOT / "experiments" / "medchange"


class LayeringTests(unittest.TestCase):
    """Code that several commands need lives in a library module (2026-10-12). A command is a module that can be run with ``python -m``
    (it has a ``__main__`` guard); no module imports a command, so a command can change without changing another."""

    @staticmethod
    def _sources():
        out = {p.stem: p for p in PACKAGE.glob("*.py") if p.name != "__init__.py"}
        out.update({f"stage1.{p.stem}": p for p in (PACKAGE / "stage1").glob("*.py") if p.name != "__init__.py"})
        return out

    @staticmethod
    def _is_command(tree):
        return any(isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                   and n.test.left.id == "__name__" for n in tree.body)

    @staticmethod
    def _imported(name, tree):
        """The pipeline modules that a module imports, relative or absolute, under their names in ``_sources``."""
        package = name.rpartition(".")[0]                     # "stage1" for a module of the subpackage, "" for the others
        found = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                base = (package + "." if package and node.level == 1 else "") + (node.module or "")
            elif (node.module or "").startswith("experiments.medchange"):
                base = node.module[len("experiments.medchange"):].lstrip(".")
            else:
                continue
            found.add(base)
            found.update((base + "." if base else "") + a.name for a in node.names)
        return found

    def _commands(self):
        return {n for n, p in self._sources().items() if self._is_command(ast.parse(text(p)))}

    def test_no_module_imports_a_command(self):
        sources = self._sources()
        commands = self._commands()
        offenders = sorted(f"{n} imports {m}" for n, p in sources.items() for m in self._imported(n, ast.parse(text(p))) if m in commands)
        self.assertEqual(offenders, [])
        self.assertGreaterEqual(len(commands), 19)

    def test_the_commands_and_the_libraries_are_those_the_readme_lists(self):
        block = text(README).split("## Repository Structure", 1)[1].split("```", 2)[1]
        commands_part = block.split("commands (", 1)[1].split("libraries (", 1)[0]
        libraries_part = block.split("libraries (", 1)[1].split("manifest.json", 1)[0]
        listed_commands = set(re.findall(r"(\w+)\.py\b", commands_part))
        listed_libraries = set(re.findall(r"(\w+)\.py\b", libraries_part))
        modules = {n for n in self._sources() if "." not in n}
        self.assertEqual(listed_commands, self._commands())
        self.assertEqual(listed_libraries, modules - self._commands())


class CodeAgreesWithTheManifestsTests(unittest.TestCase):

    def test_the_places_that_hold_the_number_of_fresh_questions_agree_with_the_manifest(self):
        manifest = json.loads(text(PACKAGE / "manifest_fresh.json"))
        self.assertEqual({analyze_fresh.EXPECTED, fresh_benchmark.N_DEFAULT, pipeline_support.EXPECTED_ITEMS["fresh"]}, {manifest["items"]})
        self.assertEqual(len(manifest["rows"]), manifest["items"])
        self.assertEqual(sum(manifest["labels"].values()), manifest["items"])


class CommittedFreshRunTests(unittest.TestCase):
    """The committed run of the pre-registered test covers exactly the registered question list: nothing is missing, repeated or extra,
    and every record was made by the arms and the generator of the registration."""

    MANIFEST = json.loads(text(PACKAGE / "manifest_fresh.json"))
    IDS = ["FR-%05d" % row for row in MANIFEST["rows"]]
    ARMS = ("R2", "R2C", "R2V")

    def test_the_registered_list_has_the_registered_hash_and_the_protocol_quotes_it(self):
        self.assertEqual(hashlib.sha256(json.dumps(self.IDS).encode()).hexdigest(), self.MANIFEST["item_ids_sha256"])
        self.assertEqual(len(set(self.IDS)), 850)
        self.assertIn(self.MANIFEST["item_ids_sha256"][:16], text(DOCS / "protocol.md"))

    def test_the_answers_are_one_per_question_and_arm(self):
        answers = load_jsonl(RESULTS / "rag2_answers_fresh.jsonl")
        self.assertEqual(len(answers), 3 * 850)
        for arm in self.ARMS:
            self.assertEqual(sorted(r["item_id"] for r in answers if r["arm"] == arm), sorted(self.IDS), arm)
        self.assertEqual({r["arm"] for r in answers}, set(self.ARMS))               # no B0, B1, R2V-ND or ablation arm was run
        self.assertTrue(all(r["verdict"] in ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION", None) for r in answers))
        self.assertEqual({r["settings"] for r in answers if r["arm"] == "R2"}, {"1743afd7045b"})         # the frozen design

    def test_the_other_steps_cover_the_same_questions(self):
        self.assertEqual(sorted(r["item_id"] for r in load_jsonl(RESULTS / "rag2_rationales_fresh.jsonl")), sorted(self.IDS))
        self.assertEqual(sorted(r["item_id"] for r in load_jsonl(RESULTS / "rag2_lists_fresh.ids.jsonl")), sorted(self.IDS))
        judged = {r["item_id"] for r in load_jsonl(RESULTS / "rag2_filter_fresh.jsonl")}
        self.assertLessEqual(judged, set(self.IDS))                                 # a question without an eligible record has nothing to judge
        self.assertGreater(len(judged), 0.8 * 850)

    def test_the_retrieval_description_is_recomputed_from_the_committed_answers(self):
        """Mean age needs the dates of the reviews, which are not committed; the rest follows from the answers alone."""
        described = json.loads(text(RESULTS / "rag2_analysis_fresh.json"))["retrieval"]["arms"]["R2"]
        by_arm = {arm: {r["item_id"]: r for r in load_jsonl(RESULTS / "rag2_answers_fresh.jsonl") if r["arm"] == arm} for arm in self.ARMS}
        for arm in ("R2C", "R2V"):
            self.assertEqual({i: r["admitted"] for i, r in by_arm[arm].items()}, {i: r["admitted"] for i, r in by_arm["R2"].items()}, arm)
        counts = [len(r["admitted"]) for r in by_arm["R2"].values()]
        self.assertEqual(round(sum(counts) / len(counts), 2), described["mean_admitted"])
        self.assertEqual(round(sum(c == 0 for c in counts) / len(counts), 6), described["share_without_evidence"])
        strata = [s for r in by_arm["R2"].values() for s in r["admitted_stratum"]]
        self.assertEqual({s: round(strata.count(s) / len(strata), 6) for s in ("SR/MA", "RCT", "other")}, described["evidence_mix"])

    def test_the_run_used_the_generator_of_the_design_and_no_judge_model(self):
        env = json.loads(text(RESULTS / "rag2_environment_fresh.json"))
        design = json.loads(text(RESULTS / "rag2_design.json"))
        self.assertEqual((env["phase"], env["judge_file"], env["tracked_code_modified"]), ("fresh", None, False))
        self.assertEqual(env["generator_file"], "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
        cfg = json.loads(text(RESULTS / "rag2_answers_fresh.config.json"))
        self.assertEqual(cfg["model_sha256"], design["generator_model_sha256"])
        self.assertEqual(cfg["model_file"], env["generator_file"])
        self.assertFalse((RESULTS / "rag2_directness_fresh.jsonl").exists())        # the fresh test has no directness judge


@unittest.skipUnless(os.environ.get("MEDCHANGE_DIR"), "set MEDCHANGE_DIR to a clone of jvladika/MedChange to rebuild the question sets from the release")
class RebuildFromTheReleaseTests(unittest.TestCase):
    """Run with ``MEDCHANGE_DIR=<clone>``: the three question sets are rebuilt from the public release and their manifests (counts, input
    hashes, item lists) must equal the committed ones byte for byte. Nothing is written outside a temporary folder."""

    def test_the_three_manifests_are_rebuilt_byte_for_byte(self):
        from experiments.medchange import ad_benchmark, build_benchmark
        release = os.environ["MEDCHANGE_DIR"]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            tmp = Path(tmp)
            data = tmp / "data"
            self.assertEqual(build_benchmark.main(["--medchange-dir", release, "--out-dir", str(data), "--manifest", str(tmp / "manifest.json")]), 0)
            self.assertEqual(ad_benchmark.main(["--medchange-dir", release, "--data-dir", str(data), "--manifest", str(tmp / "manifest_ad.json")]), 0)
            self.assertEqual(fresh_benchmark.main(["--medchange-dir", release, "--data-dir", str(data), "--manifest", str(tmp / "manifest_fresh.json")]), 0)
            for name in ("manifest.json", "manifest_ad.json", "manifest_fresh.json"):
                rebuilt = (tmp / name).read_bytes().replace(b"\r\n", b"\n")
                committed = (PACKAGE / name).read_bytes().replace(b"\r\n", b"\n")           # a Windows checkout may convert line endings
                self.assertEqual(rebuilt, committed, name)


class DocumentationIntegrityTests(unittest.TestCase):
    """Every repository path and every ``python -m`` command that the current documents name must exist.
    Dangling references were the most common defect found in the audits; this makes them fail loudly."""

    PATH = re.compile(r"`((?:src|evaluation|tests|experiments|corpus|docs|_archive)/[^`\s]*)`")
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
                "freeze_candidates", "generate_answers", "label_audit", "fresh_benchmark", "fresh_supply", "analyze_fresh", "ad_subgroup",
                "class_balance", "ad_cochrane_supply", "ad_case_study", "headroom", "verdict_consistency")]:
            tree = ast.parse(text(ROOT / "experiments" / package / f"{module}.py"))
            flags[(package, module)] = {a.value for node in ast.walk(tree)
                                        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                                        for a in node.args if isinstance(a, ast.Constant) and str(a.value).startswith("--")}
        missing = []
        for rel, body in self._docs():
            for line in body.splitlines():
                m = re.search(r"python -m experiments\.(medchange)\.(\w+)(.*)", line)
                if not m or (m.group(1), m.group(2)) not in flags:
                    continue
                command = re.split(r"`|\s\|\s", m.group(3))[0]
                for flag in re.findall(r"(?<![\w-])(--[a-z][\w-]*)", command):
                    if flag not in flags[(m.group(1), m.group(2))]:
                        missing.append(f"{rel}: {m.group(2)} {flag}")
        self.assertEqual(missing, [])

    #: A repository file cited as ``dir/.../name.ext`` in a comment or docstring.
    CODE_PATH = re.compile(
        r"(?<![\w./-])((?:src|evaluation|tests|experiments|corpus|docs|_archive)/[\w./-]*\w"
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
        r"(?<![\w./-])(?:experiments/shared|experiments/results|src/(?:baseline|proposed|common|temporal_filter)"
        r"|evaluation/(?:runner|freezing|questions|accuracy|rag_metrics|annotation|stats)\.py"
        r"|experiments/medchange/runner\.py|RAG2_FINDINGS\.md"
        r"|evaluation/tests"
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
