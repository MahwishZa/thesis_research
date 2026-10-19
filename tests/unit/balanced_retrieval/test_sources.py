"""Tests of how the four corpora are fetched: balanced_retrieval.scope, balanced_retrieval.ncbi, balanced_retrieval.documents,
balanced_retrieval.pubmed, balanced_retrieval.pmc, balanced_retrieval.guidelines and balanced_retrieval.textbooks.

No test reaches the internet. The service is a stand-in that answers like E-utilities, the XML is written the way PubMed and
PMC write it, and the hub is a stand-in with the calls that ``huggingface_hub`` offers.
"""

import io
import json
import unittest
import urllib.error
from datetime import date
from urllib.parse import parse_qs

from balanced_retrieval import (
    documents,
    guidelines,
    ncbi,
    pmc,
    pubmed,
    scope,
    textbooks,
)
from balanced_retrieval.cli import main
from rationale_query.cli import Streams
from rationale_query.store import ConfigMismatch, StoreError
from tests.support import TempDirTestCase, np

PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation>
   <PMID Version="1">111</PMID>
   <Article>
    <Journal><JournalIssue><PubDate><Year>2020</Year><Month>Mar</Month><Day>5</Day></PubDate></JournalIssue></Journal>
    <ArticleTitle>Donepezil in <i>mild</i> Alzheimer disease.</ArticleTitle>
    <Abstract>
     <AbstractText Label="BACKGROUND">Cholinesterase inhibitors are used.</AbstractText>
     <AbstractText Label="RESULTS">Memory improved   slightly.</AbstractText>
    </Abstract>
    <PublicationTypeList><PublicationType UI="D016428">Journal Article</PublicationType></PublicationTypeList>
   </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList>
   <ArticleId IdType="pubmed">111</ArticleId><ArticleId IdType="doi">10.1/abc</ArticleId><ArticleId IdType="pmc">PMC555</ArticleId>
  </ArticleIdList></PubmedData>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation><PMID>112</PMID><Article><ArticleTitle>No abstract here</ArticleTitle></Article></MedlineCitation>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation><PMID>113</PMID><Article>
   <ArticleDate DateType="Electronic"><Year>2019</Year><Month>11</Month><Day>02</Day></ArticleDate>
   <ArticleTitle>A withdrawn study</ArticleTitle><Abstract><AbstractText>Text.</AbstractText></Abstract>
   <PublicationTypeList><PublicationType>Retracted Publication</PublicationType></PublicationTypeList>
  </Article></MedlineCitation>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation><PMID>114</PMID><Article>
   <Journal><JournalIssue><PubDate><Year>2015</Year></PubDate></JournalIssue></Journal>
   <ArticleTitle>Plain abstract</ArticleTitle><Abstract><AbstractText>Single block.</AbstractText></Abstract>
  </Article></MedlineCitation>
 </PubmedArticle>
</PubmedArticleSet>
"""

PMC_XML = """<?xml version="1.0"?>
<pmc-articleset>
 <article xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:mml="http://www.w3.org/1998/Math/MathML" xml:lang="en">
  <front><article-meta>
   <article-id pub-id-type="pmid">111</article-id><article-id pub-id-type="pmc">555</article-id><article-id pub-id-type="doi">10.1/abc</article-id>
   <title-group><article-title>Donepezil in mild Alzheimer disease</article-title></title-group>
   <abstract><p>Short abstract.</p></abstract>
   <abstract abstract-type="graphical"><p>GRAPHICAL TEXT</p></abstract>
   <permissions><license license-type="open-access" xlink:href="https://creativecommons.org/licenses/by-nc/4.0/"><license-p>Open access.</license-p></license></permissions>
  </article-meta></front>
  <body>
   <sec><title>Introduction</title><p>Dementia is common<xref ref-type="bibr" rid="b1">1</xref> and costly.</p>
    <table-wrap><caption><p>TABLE TEXT</p></caption></table-wrap>
    <fig><caption><p>FIGURE TEXT</p></caption></fig>
    <sec><title>Aims</title><p>We tested <inline-formula><mml:math><mml:mi>x</mml:mi></mml:math></inline-formula> donepezil.</p></sec>
   </sec>
   <sec><title>Methods</title><list><list-item><p>Step one.</p></list-item></list></sec>
  </body>
  <back><ref-list><ref><mixed-citation>REFERENCE TEXT</mixed-citation></ref></ref-list></back>
 </article>
 <article xml:lang="en"><front><article-meta><article-id pub-id-type="pmc">777</article-id>
  <title-group><article-title>Not open access</article-title></title-group></article-meta></front></article>
 <article xml:lang="fr"><front><article-meta><article-id pub-id-type="pmc">888</article-id></article-meta></front><body><sec><p>Texte.</p></sec></body></article>
 <article xml:lang="en"><front><article-meta><article-id pub-id-type="pmc">999</article-id>
  <title-group><article-title>No PubMed number</article-title></title-group></article-meta></front><body><p>Only body.</p></body></article>
</pmc-articleset>
"""


class ScopeTests(unittest.TestCase):
    def test_the_scope_names_alzheimer_disease_and_asks_for_an_abstract_in_english(self):
        query = scope.pubmed_scope()
        self.assertIn('"Alzheimer Disease"[Mesh]', query)
        self.assertTrue(query.endswith("AND English[lang] AND hasabstract"))
        self.assertEqual(query.count("("), query.count(")"))

    def test_the_neighbouring_conditions_are_only_taken_together_with_dementia_or_cognition(self):
        query = scope.pubmed_scope()
        self.assertIn('"Delirium"[Mesh]', query)
        self.assertRegex(query, r'"Delirium"\[Mesh\].*?\) AND \(.*?dementia\[tiab\]')


class FakeService:
    """Answers esearch and efetch from a list of (id, 'YYYY-MM-DD') records."""

    def __init__(self, records, fail_first=0, status=503):
        self.records = records
        self.calls = []
        self.fail = fail_first
        self.status = status

    def __call__(self, url, data):
        self.calls.append((url.rsplit("/", 1)[1], parse_qs(data.decode())))
        if self.fail:
            self.fail -= 1
            raise urllib.error.HTTPError(url, self.status, "busy", {}, None)
        name, params = self.calls[-1]
        if name == "esearch.fcgi":
            low, high = params["mindate"][0].replace("/", "-"), params["maxdate"][0].replace("/", "-")
            hits = [i for i, d in sorted(self.records, key=lambda r: r[1], reverse=params["sort"] == ["pub_date"]) if low <= d <= high]
            ids = hits[: int(params["retmax"][0])]
            return json.dumps({"esearchresult": {"count": str(len(hits)), "idlist": ids}}).encode()
        return b"<ok/>"


def quiet_client(service, **options):
    clock = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    client = ncbi.Client(api_key="", email="", opener=service, sleep=sleep, clock=lambda: clock[0], **options)
    client.sleeps = sleeps
    return client


class NcbiClientTests(unittest.TestCase):
    def test_requests_are_spaced_to_three_a_second_without_a_key_and_ten_with_one(self):
        client = quiet_client(FakeService([]))
        for _ in range(3):
            client.call("einfo.fcgi")
        self.assertAlmostEqual(sum(client.sleeps), 2 * 0.34, places=6)
        keyed = ncbi.Client(api_key="k", email="", opener=FakeService([]), sleep=lambda s: None)
        self.assertLess(keyed.interval, 0.12)

    def test_the_tool_name_the_key_and_the_email_go_with_every_request(self):
        service = FakeService([])
        ncbi.Client(api_key="KEY", email="a@b.c", opener=service, sleep=lambda s: None).call("einfo.fcgi", db="pubmed")
        params = service.calls[0][1]
        self.assertEqual((params["tool"], params["api_key"], params["email"], params["db"]), (["thesis_research"], ["KEY"], ["a@b.c"], ["pubmed"]))

    def test_a_busy_service_is_asked_again_after_a_growing_pause(self):
        service = FakeService([], fail_first=2)
        client = quiet_client(service)
        self.assertEqual(client.call("einfo.fcgi"), b"<ok/>")
        self.assertEqual(len(service.calls), 3)
        self.assertEqual([s for s in client.sleeps if s >= 2], [2.0, 4.0])

    def test_a_connection_that_drops_in_the_middle_of_an_answer_is_tried_again(self):
        import http.client

        answers = [http.client.IncompleteRead(b"half"), ConnectionResetError("reset"), b"<ok/>"]

        def opener(url, data):
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        client = quiet_client(opener)
        self.assertEqual(client.call("efetch.fcgi"), b"<ok/>")
        self.assertEqual(answers, [])

    def test_a_request_that_keeps_failing_ends_in_an_error_that_says_so(self):
        with self.assertRaisesRegex(ncbi.NcbiError, "503"):
            quiet_client(FakeService([], fail_first=99), retries=2).call("einfo.fcgi")

    def test_a_refusal_that_will_not_pass_is_not_repeated(self):
        service = FakeService([], fail_first=99, status=400)
        with self.assertRaisesRegex(ncbi.NcbiError, "400") as caught:
            quiet_client(service).call("einfo.fcgi")
        self.assertEqual(len(service.calls), 1)
        self.assertEqual(caught.exception.status, 400)


class FetchIsolatingTests(unittest.TestCase):
    """A list that the service refuses with 400 is split until the one identifier it dislikes stands alone."""

    def make(self, bad):
        calls = []

        def fetch(part):
            calls.append(list(part))
            if any(i in bad for i in part):
                raise ncbi.NcbiError("refused", status=400)
            return [f"doc{i}" for i in part]

        return fetch, calls

    def test_the_other_identifiers_are_fetched_and_the_refused_one_is_listed(self):
        fetch, calls = self.make({"7"})
        rejected = []
        found = ncbi.fetch_isolating(quiet_client(FakeService([])), [str(i) for i in range(1, 11)], fetch, rejected)
        self.assertEqual(sorted(found), sorted(f"doc{i}" for i in range(1, 11) if i != 7))
        self.assertEqual(rejected, ["7"])
        self.assertLess(len(calls), 20)  # a few extra requests, not one per identifier

    def test_a_refused_identifier_is_asked_for_once_more_after_a_pause(self):
        fetch, calls = self.make({"3"})
        client = quiet_client(FakeService([]))
        ncbi.fetch_isolating(client, ["3"], fetch, [])
        self.assertEqual(calls, [["3"], ["3"]])
        self.assertIn(5.0, client.sleeps)

    def test_a_refusal_that_was_only_a_glitch_loses_nothing(self):
        state = {"first": True}

        def fetch(part):
            if state["first"]:
                state["first"] = False
                raise ncbi.NcbiError("glitch", status=400)
            return [f"doc{i}" for i in part]

        rejected = []
        found = ncbi.fetch_isolating(quiet_client(FakeService([])), ["1", "2", "3"], fetch, rejected)
        self.assertEqual((sorted(found), rejected), (["doc1", "doc2", "doc3"], []))

    def test_when_every_identifier_of_a_list_is_refused_the_request_itself_is_wrong(self):
        fetch, _ = self.make({"1", "2", "3", "4", "5"})
        with self.assertRaisesRegex(ncbi.NcbiError, "refused"):
            ncbi.fetch_isolating(quiet_client(FakeService([])), ["1", "2", "3", "4", "5"], fetch, [])

    def test_other_errors_are_not_swallowed(self):
        def fetch(part):
            raise ncbi.NcbiError("down", status=503)

        with self.assertRaisesRegex(ncbi.NcbiError, "down"):
            ncbi.fetch_isolating(quiet_client(FakeService([])), ["1", "2"], fetch, [])


class SearchIdsTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, ncbi, "MAX_HITS", ncbi.MAX_HITS)
        ncbi.MAX_HITS = 4

    def test_a_window_with_too_many_hits_is_cut_until_every_hit_is_returned_once(self):
        records = [(str(1000 + n), f"20{10 + n // 12:02d}-{n % 12 + 1:02d}-15") for n in range(40)]
        service = FakeService(records)
        found = ncbi.search_ids(quiet_client(service), "pubmed", "x", date(2030, 1, 1))
        self.assertEqual(sorted(found), sorted(i for i, _ in records))
        self.assertEqual(len(found), len(set(found)))

    def test_the_newest_window_comes_first_so_that_a_trial_run_holds_recent_records(self):
        records = [(str(n), f"20{10 + n}-06-15") for n in range(10)]
        found = ncbi.search_ids(quiet_client(FakeService(records)), "pubmed", "x", date(2030, 1, 1))
        self.assertEqual(found, [str(n) for n in range(9, -1, -1)])

    def test_the_number_found_so_far_is_reported_after_every_window(self):
        records = [(str(n), f"20{10 + n}-06-15") for n in range(10)]
        reported = []
        ncbi.search_ids(quiet_client(FakeService(records)), "pubmed", "x", date(2030, 1, 1), reported.append)
        self.assertEqual(reported[-1], 10)
        self.assertEqual(reported, sorted(reported))

    def test_only_records_up_to_the_date_are_found(self):
        records = [("1", "2019-05-01"), ("2", "2020-05-01"), ("3", "2021-05-01")]
        found = ncbi.search_ids(quiet_client(FakeService(records)), "pubmed", "x", date(2020, 12, 31))
        self.assertEqual(sorted(found), ["1", "2"])

    def test_the_count_of_a_scope_is_one_request(self):
        service = FakeService([("1", "2019-05-01"), ("2", "2020-05-01")])
        self.assertEqual(ncbi.count_hits(quiet_client(service), "pubmed", "x", date(2030, 1, 1)), 2)
        self.assertEqual(len(service.calls), 1)

    def test_a_single_day_with_too_many_hits_is_an_error_not_a_silent_loss(self):
        records = [(str(n), "2020-01-01") for n in range(10)]
        with self.assertRaisesRegex(ncbi.NcbiError, "single day"):
            ncbi.search_ids(quiet_client(FakeService(records)), "pubmed", "x", date(2021, 1, 1))

    def test_a_search_the_service_refuses_is_an_error_with_its_message(self):
        def refuse(url, data):
            return json.dumps({"esearchresult": {"ERROR": "Invalid query"}}).encode()

        with self.assertRaisesRegex(ncbi.NcbiError, "Invalid query"):
            quiet_client(refuse).search("pubmed", "((", date(2000, 1, 1), date(2001, 1, 1))


class ParsePubmedTests(unittest.TestCase):
    def setUp(self):
        self.documents = pubmed.parse_pubmed(PUBMED_XML)

    def test_records_without_an_abstract_and_retracted_ones_are_left_out(self):
        self.assertEqual([d["pmid"] for d in self.documents], ["111", "114"])

    def test_a_structured_abstract_keeps_its_labels_and_the_title_comes_first(self):
        self.assertEqual(
            self.documents[0]["text"],
            "Donepezil in mild Alzheimer disease. BACKGROUND: Cholinesterase inhibitors are used. RESULTS: Memory improved slightly.",
        )

    def test_the_key_the_identifiers_and_the_date_are_kept(self):
        first = self.documents[0]
        self.assertEqual((first["doc_key"], first["doi"], first["pmcid"], first["date"]), ("PMID:111", "10.1/abc", "PMC555", "2020-03-05"))
        self.assertEqual((self.documents[1]["doi"], self.documents[1]["pmcid"], self.documents[1]["date"]), ("", "", "2015"))

    def test_the_records_that_were_not_kept_are_counted_by_reason(self):
        from collections import Counter

        left_out = Counter()
        pubmed.parse_pubmed(PUBMED_XML, left_out)
        self.assertEqual(dict(left_out), {"no abstract": 1, "retracted": 1})

    def test_a_broken_answer_is_an_error(self):
        with self.assertRaises(StoreError):
            pubmed.parse_pubmed("<PubmedArticleSet><oops>")


class ParsePmcTests(unittest.TestCase):
    def setUp(self):
        self.documents = pmc.parse_pmc(PMC_XML)

    def test_only_english_articles_with_a_body_are_kept(self):
        self.assertEqual([d["pmcid"] for d in self.documents], ["PMC555", "PMC999"])

    def test_tables_figures_formulas_references_and_citation_marks_are_left_out(self):
        text = self.documents[0]["text"]
        for noise in ("TABLE TEXT", "FIGURE TEXT", "REFERENCE TEXT", "GRAPHICAL TEXT"):
            self.assertNotIn(noise, text)
        self.assertIn("Dementia is common and costly.", text)
        self.assertIn("We tested donepezil.", text.replace("  ", " "))

    def test_title_abstract_headings_and_paragraphs_come_in_reading_order(self):
        lines = self.documents[0]["text"].split("\n")
        self.assertEqual(lines[:4], ["Donepezil in mild Alzheimer disease", "Short abstract.", "Introduction", "Dementia is common and costly."])
        self.assertIn("Step one.", lines)

    def test_the_key_is_the_pubmed_number_when_there_is_one_else_the_pmc_number(self):
        self.assertEqual([d["doc_key"] for d in self.documents], ["PMID:111", "PMCID:PMC999"])

    def test_the_articles_that_were_not_kept_are_counted_by_reason(self):
        from collections import Counter

        left_out = Counter()
        pmc.parse_pmc(PMC_XML, left_out)
        self.assertEqual(dict(left_out), {"no full text (not open access, or a scan)": 1, "not English": 1})

    def test_the_licence_group_is_read_from_the_licence_statement(self):
        self.assertEqual([d["licence"] for d in self.documents], ["cc-by-nc", "none"])

    def test_other_statements_give_other_and_the_public_domain_gives_cc0(self):
        template = '<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta>%s</article-meta></front><body><p>x</p></body></article>'
        for statement, group in (
            ('<permissions><license xlink:href="https://creativecommons.org/publicdomain/zero/1.0/"/></permissions>', "cc0"),
            ('<permissions><license><license-p>All rights reserved</license-p></license></permissions>', "other"),
            ('<permissions><license><license-p>CC BY: https://creativecommons.org/licenses/by/4.0/ applies</license-p></license></permissions>', "cc-by"),
        ):
            with self.subTest(group=group):
                self.assertEqual(pmc.licence_group(__import__("xml.etree.ElementTree", fromlist=["x"]).fromstring(template % statement)), group)


class DownloadInBatchesTests(TempDirTestCase):
    def run_download(self, out, ids, batch=2, settings=None, fail_at=None, limit=None):
        calls = []

        def fetch(batch_ids):
            if fail_at is not None and len(calls) == fail_at:
                raise RuntimeError("the connection dropped")
            calls.append(batch_ids)
            return [{"doc_key": f"PMID:{i}", "text": f"text {i}"} for i in batch_ids]

        state = documents.download_in_batches(out, settings or {"s": 1}, lambda: ids, fetch, batch_size=batch, limit=limit)
        return state, calls

    def test_every_identifier_is_fetched_in_batches_and_written_in_order(self):
        out = self.dir / "docs.jsonl"
        state, calls = self.run_download(out, ["1", "2", "3", "4", "5"])
        self.assertEqual(calls, [["1", "2"], ["3", "4"], ["5"]])
        self.assertEqual((state["done"], state["ids"], state["documents"]), (5, 5, 5))
        self.assertEqual([json.loads(line)["doc_key"] for line in out.read_text().splitlines()], [f"PMID:{i}" for i in range(1, 6)])

    def test_a_run_that_stops_continues_after_the_last_finished_batch(self):
        out = self.dir / "docs.jsonl"
        with self.assertRaises(RuntimeError):
            self.run_download(out, ["1", "2", "3", "4", "5"], fail_at=2)
        state = documents.read_state(out)
        self.assertEqual(state["done"], 4)
        state, calls = self.run_download(out, ["never", "used"])
        self.assertEqual(calls, [["5"]])
        self.assertEqual(len(out.read_text().splitlines()), 5)

    def test_bytes_written_after_the_last_finished_batch_are_cut_off(self):
        out = self.dir / "docs.jsonl"
        self.run_download(out, ["1", "2", "3", "4"], limit=2)
        with open(out, "ab") as handle:
            handle.write(b'{"doc_key":"half')
        self.run_download(out, [])
        self.assertEqual([json.loads(line)["doc_key"] for line in out.read_text().splitlines()], [f"PMID:{i}" for i in range(1, 5)])

    def test_a_limit_stops_early_and_a_later_run_without_it_finishes(self):
        out = self.dir / "docs.jsonl"
        state, _ = self.run_download(out, ["1", "2", "3", "4"], limit=3)
        self.assertEqual(state["done"], 3)
        state, calls = self.run_download(out, [])
        self.assertEqual((state["done"], calls), (4, [["4"]]))

    def test_a_download_with_other_settings_is_refused(self):
        out = self.dir / "docs.jsonl"
        self.run_download(out, ["1"], settings={"query": "a"})
        with self.assertRaises(ConfigMismatch):
            self.run_download(out, ["1"], settings={"query": "b"})

    def test_a_file_without_a_state_is_never_appended_to(self):
        out = self.write("docs.jsonl", "old\n")
        with self.assertRaises(StoreError):
            self.run_download(out, ["1"])
        self.assertEqual(out.read_text(), "old\n")

    def test_the_reasons_a_batch_gives_for_leaving_documents_out_are_added_up_in_the_state(self):
        out = self.dir / "docs.jsonl"
        state = documents.download_in_batches(out, {}, lambda: ["1", "2", "3"], lambda ids: ([], {"no abstract": len(ids)}), batch_size=2)
        self.assertEqual(state["notes"], {"no abstract": 3})

    def test_a_batch_may_return_fewer_documents_than_identifiers(self):
        out = self.dir / "docs.jsonl"
        state = documents.download_in_batches(out, {}, lambda: ["1", "2"], lambda ids: [], batch_size=5)
        self.assertEqual((state["done"], state["documents"]), (2, 0))


class FetchPubmedAndPmcTests(TempDirTestCase):
    def test_pubmed_is_searched_as_of_a_date_and_fetched_in_batches_of_200(self):
        records = [(str(n), "2020-01-01") for n in range(1, 451)]

        class Service(FakeService):
            def __call__(self, url, data):
                if url.endswith("efetch.fcgi"):
                    self.calls.append(("efetch.fcgi", parse_qs(data.decode())))
                    return PUBMED_XML.encode()
                return super().__call__(url, data)

        service = Service(records)
        out = self.dir / "pubmed.jsonl"
        state = pubmed.fetch_pubmed(quiet_client(service), out, as_of=date(2026, 9, 13))
        efetches = [p for name, p in service.calls if name == "efetch.fcgi"]
        self.assertEqual([len(p["id"][0].split(",")) for p in efetches], [200, 200, 50])
        self.assertEqual(efetches[0]["db"], ["pubmed"])
        self.assertEqual(state["ids"], 450)
        self.assertEqual(documents.read_state(out)["settings"], {"source": "pubmed", "query": scope.pubmed_scope(), "as_of": "2026-09-13"})

    def test_identifiers_the_service_does_not_return_are_counted_as_left_out(self):
        out = self.dir / "pubmed.jsonl"
        service = FakeService([(str(n), "2020-01-01") for n in range(1, 21)])
        original = service.__call__

        def answering(url, data):
            return PUBMED_XML.encode() if url.endswith("efetch.fcgi") else original(url, data)

        pubmed.fetch_pubmed(quiet_client(answering), out, as_of=date(2026, 9, 13))
        notes = documents.read_state(out)["notes"]
        self.assertEqual(notes["no abstract"], 1)
        self.assertEqual(notes["retracted"], 1)
        self.assertEqual(notes["not returned, or not a journal article"], 20 - 2 - 2)

    def test_a_pmc_number_the_service_refuses_is_left_out_and_listed_and_the_download_goes_on(self):
        pubmed_file = self.write("pubmed.jsonl", "".join(json.dumps({"pmcid": f"PMC{n}"}) + "\n" for n in (5, 6, 666, 8, 9, 10, 11)))

        def opener(url, data):
            if "666" in parse_qs(data.decode())["id"][0].split(","):
                raise urllib.error.HTTPError(url, 400, "Bad Request", {}, None)
            return PMC_XML.encode()

        out = self.dir / "pmc.jsonl"
        state = pmc.fetch_pmc(quiet_client(opener), out, pubmed_file=pubmed_file)
        self.assertEqual(state["done"], state["ids"])
        self.assertEqual(state["notes"]["rejected by the service (400 Bad Request)"], 1)
        self.assertEqual((self.dir / "pmc.jsonl.rejected.txt").read_text().split(), ["666"])

    def test_pmc_fetches_the_pmc_numbers_of_the_pubmed_download_without_repeats(self):
        pubmed_file = self.write(
            "pubmed.jsonl",
            "".join(json.dumps(d) + "\n" for d in [{"pmcid": "PMC5"}, {"pmcid": ""}, {"pmcid": "PMC7"}, {"pmcid": "PMC5"}, {}]),
        )
        self.assertEqual(pmc.pmc_ids_of(pubmed_file), ["5", "7"])
        seen = []

        def opener(url, data):
            seen.append(parse_qs(data.decode()))
            return PMC_XML.encode()

        out = self.dir / "pmc.jsonl"
        state = pmc.fetch_pmc(quiet_client(opener), out, pubmed_file=pubmed_file)
        self.assertEqual((seen[0]["db"], seen[0]["id"]), (["pmc"], ["5,7"]))
        self.assertEqual((state["ids"], state["documents"]), (2, 2))


class GuidelinesTests(TempDirTestCase):
    LINES = (
        json.dumps({"id": "g1", "source": "nice", "title": "Dementia", "clean_text": "Offer donepezil.", "raw_text": "RAW", "url": "https://x/1"}),
        json.dumps({"id": "g2", "source": "cdc", "title": "", "raw_text": "Only raw."}),
        json.dumps({"id": "g3", "source": "who", "title": "Empty", "clean_text": "   "}),
        "",
    )

    def test_a_guideline_becomes_a_document_keyed_by_its_address_else_its_id(self):
        found = list(guidelines.read_guidelines(self.LINES))
        self.assertEqual([d["doc_key"] for d in found], ["URL:https://x/1", "GUIDELINE:g2"])
        self.assertEqual(found[0]["text"], "Dementia\nOffer donepezil.")
        self.assertEqual((found[1]["text"], found[1]["source"]), ("Only raw.", "cdc"))

    def test_the_open_file_with_the_newest_version_is_chosen(self):
        names = ["README.md", "guidelines.jsonl", "open_guidelines.jsonl", "open_guidelines_v2.jsonl"]
        self.assertEqual(guidelines.choose_file(names), "open_guidelines_v2.jsonl")
        with self.assertRaises(StoreError):
            guidelines.choose_file(["guidelines.jsonl"])

    def test_the_download_records_the_file_and_the_revision(self):
        source = self.write("hub/open.jsonl", "\n".join(self.LINES))

        class Hub:
            class HfApi:
                def list_repo_files(self, repo, **kw):
                    return ["open.jsonl"]

                def dataset_info(self, repo, **kw):
                    return type("Info", (), {"sha": "abc123"})()

            @staticmethod
            def hf_hub_download(repo, name, **kw):
                return str(source)

        out = self.dir / "guidelines.jsonl"
        info = guidelines.download_guidelines(out, hub=Hub)
        self.assertEqual(info, {"source": "epfl-llm/guidelines/open.jsonl", "revision": "abc123", "documents": 2})
        self.assertEqual(len(out.read_text().splitlines()), 2)
        self.assertEqual(json.loads((self.dir / "guidelines.jsonl.source.json").read_text())["revision"], "abc123")


class TextbooksTests(TempDirTestCase):
    def test_pieces_that_do_not_overlap_are_joined_with_a_space(self):
        self.assertEqual(textbooks.join_book(["First piece.", " Second piece. ", ""]), "First piece. Second piece.")

    def test_a_stretch_shared_by_two_neighbouring_pieces_is_kept_once(self):
        first = "The hippocampus is affected early in Alzheimer disease and atrophy follows"
        second = "Alzheimer disease and atrophy follows. Memantine is used later."
        self.assertEqual(
            textbooks.join_book([first, second]),
            "The hippocampus is affected early in Alzheimer disease and atrophy follows. Memantine is used later.",
        )

    def test_a_short_coincidence_is_not_taken_for_an_overlap(self):
        self.assertEqual(textbooks.join_book(["ends with the", "the next begins"]), "ends with the the next begins")

    def test_every_book_is_one_document_keyed_by_its_file_name(self):
        files = {
            "chunk/Neurology_Adams.jsonl": [json.dumps({"id": "Neurology_Adams_0", "content": "Part one."}), json.dumps({"id": "Neurology_Adams_1", "contents": "Part two."})],
            "chunk/Anatomy_Gray.jsonl": [json.dumps({"id": "Anatomy_Gray_0", "content": "Bones."})],
            "chunk/Empty.jsonl": ["", json.dumps({"content": "  "})],
        }
        found = textbooks.book_documents(files)
        self.assertEqual([d["doc_key"] for d in found], ["BOOK:Anatomy_Gray", "BOOK:Neurology_Adams"])
        self.assertEqual(found[1]["text"], "Part one. Part two.")

    def test_the_download_reads_every_chunk_file_of_the_dataset(self):
        self.write("hub/chunk/A.jsonl", json.dumps({"content": "Alpha."}) + "\n")
        self.write("hub/chunk/B.jsonl", json.dumps({"content": "Beta."}) + "\n")
        root = self.dir

        class Hub:
            class HfApi:
                def list_repo_files(self, repo, **kw):
                    return ["README.md", "chunk/B.jsonl", "chunk/A.jsonl"]

                def dataset_info(self, repo, **kw):
                    return type("Info", (), {"sha": "def456"})()

            @staticmethod
            def hf_hub_download(repo, name, **kw):
                return str(root / "hub" / name)

        out = self.dir / "textbooks.jsonl"
        info = textbooks.download_textbooks(out, hub=Hub)
        self.assertEqual((info["documents"], info["revision"]), (2, "def456"))
        self.assertEqual([json.loads(line)["doc_key"] for line in out.read_text().splitlines()], ["BOOK:A", "BOOK:B"])


class FetchCommandTests(TempDirTestCase):
    def run_cli(self, argv, service=None):
        out, err = io.StringIO(), io.StringIO()
        status = main(argv, streams=Streams(out, err), client_factory=(lambda: quiet_client(service)) if service else None)
        return status, out.getvalue(), err.getvalue()

    def test_count_only_reports_the_size_of_the_scope_without_downloading(self):
        service = FakeService([("1", "2019-01-01"), ("2", "2020-01-01")])
        status, out, _ = self.run_cli(["fetch", "pubmed", "--as-of", "2026-09-13", "--count-only"], service)
        self.assertEqual(status, 0)
        self.assertIn("finds 2 records up to 2026-09-13", out)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_a_download_without_an_output_file_is_refused(self):
        status, _, err = self.run_cli(["fetch", "pubmed", "--as-of", "2026-09-13"], FakeService([]))
        self.assertEqual(status, 1)
        self.assertIn("--out", err)

    def test_pubmed_needs_a_date_and_pmc_needs_a_pubmed_file(self):
        for argv, words in (
            (["fetch", "pubmed", "--out", str(self.dir / "p.jsonl")], "--as-of"),
            (["fetch", "pmc", "--out", str(self.dir / "c.jsonl")], "--pubmed-file"),
        ):
            with self.subTest(source=argv[1]):
                status, _, err = self.run_cli(argv, FakeService([]))
                self.assertEqual(status, 1)
                self.assertIn(words, err)

    def test_a_download_that_was_not_finished_is_not_chunked_or_given_full_text_unless_allowed(self):
        pubmed_file = self.dir / "p.jsonl"
        documents.download_in_batches(pubmed_file, {}, lambda: ["1", "2", "3"], lambda ids: [{"doc_key": f"PMID:{i}", "text": "t", "pmcid": f"PMC{i}"} for i in ids], batch_size=1, limit=1)
        for argv in (
            ["chunk", "--corpus", "pubmed", "--documents", str(pubmed_file), "--root", str(self.dir / "corpora")],
            ["fetch", "pmc", "--pubmed-file", str(pubmed_file), "--out", str(self.dir / "c.jsonl")],
        ):
            with self.subTest(command=argv[0]):
                status, _, err = self.run_cli(argv, FakeService([]))
                self.assertEqual(status, 1)
                self.assertIn("holds only 1 of 3 identifiers", err)
                self.assertIn("--allow-partial", err)

    @unittest.skipIf(np is None, "numpy is not installed")
    def test_allow_partial_lets_a_trial_through(self):
        pubmed_file = self.dir / "p.jsonl"
        documents.download_in_batches(pubmed_file, {}, lambda: ["1", "2", "3"], lambda ids: [{"doc_key": f"PMID:{i}", "text": "some words", "pmcid": ""} for i in ids], batch_size=1, limit=1)
        status, _, err = self.run_cli(["chunk", "--corpus", "pubmed", "--documents", str(pubmed_file), "--root", str(self.dir / "corpora"), "--allow-partial"])
        self.assertEqual(status, 0, err)

    def test_a_trial_download_stops_at_the_limit_and_says_how_to_continue(self):
        class Service(FakeService):
            def __call__(self, url, data):
                if url.endswith("efetch.fcgi"):
                    return PUBMED_XML.encode()
                return super().__call__(url, data)

        service = Service([(str(n), "2020-01-01") for n in range(1, 11)])
        status, out, _ = self.run_cli(["fetch", "pubmed", "--out", str(self.dir / "p.jsonl"), "--as-of", "2026-09-13", "--limit", "4", "--quiet"], service)
        self.assertEqual(status, 0)
        self.assertIn("2 documents from 4 of 10 identifiers", out)
        self.assertIn("not finished; run the same command again", out)


if __name__ == "__main__":
    unittest.main()
