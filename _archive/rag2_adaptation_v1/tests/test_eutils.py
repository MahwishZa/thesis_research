"""The E-utilities client (a fake opener in place of the network) and the bounds on the date a PubMed record first became public."""

import http.client
import unittest

from experiments.medchange.eutils import EUtils, availability, date_bounds


class DateTests(unittest.TestCase):
    def test_date_bounds(self):
        self.assertEqual(date_bounds("2012 Oct 17"), ("2012-10-17", "2012-10-17"))
        self.assertEqual(date_bounds("2012 Oct"), ("2012-10-01", "2012-10-31"))
        self.assertEqual(date_bounds("2012 Dec"), ("2012-12-01", "2012-12-31"))
        self.assertEqual(date_bounds("2012 Oct-Dec"), ("2012-10-01", "2012-12-31"))
        self.assertEqual(date_bounds("2012"), ("2012-01-01", "2012-12-31"))
        # forms seen in PubMed that crashed or mis-parsed a first version
        self.assertEqual(date_bounds("2012 Dec-Jan"), ("2012-12-01", "2013-12-31"))
        self.assertEqual(date_bounds("2012 Winter"), ("2012-01-01", "2012-12-31"))
        self.assertEqual(date_bounds("2012 Fall-Winter"), ("2012-01-01", "2013-12-31"))
        self.assertEqual(date_bounds("2012 Oct-Spr"), ("2012-01-01", "2013-12-31"))
        self.assertEqual(date_bounds("2012 Oct 1-5"), ("2012-10-01", "2012-10-05"))
        self.assertIsNone(date_bounds("2012 Feb 30"))
        self.assertIsNone(date_bounds(None))
        self.assertIsNone(date_bounds("n/a"))
        self.assertIsNone(date_bounds(""))

    def test_availability_is_the_earliest_known_date(self):
        self.assertEqual(availability("2013 Jan", "2012 Sep 10", None),
                         {"lower": "2012-09-10", "upper": "2012-09-10"})
        self.assertEqual(availability("2013 Jan", "", None),
                         {"lower": "2013-01-01", "upper": "2013-01-31"})
        self.assertEqual(availability("", "", "2011-04-02"),
                         {"lower": "2011-04-02", "upper": "2011-04-02"})
        self.assertIsNone(availability("", "", None))


class GetTests(unittest.TestCase):
    def test_eutils_retries_a_reset_connection_and_gives_up_after_five_attempts(self):
        calls = []

        def flaky(url):
            calls.append(url)
            if len(calls) < 3:
                raise ConnectionResetError("reset by peer")            # an OSError that is neither a URLError nor a timeout
            return b'{"esearchresult": {"count": "0", "idlist": []}}'

        self.assertEqual(EUtils(opener=flaky, sleep=lambda s: None).search("x", "2020/01/01", 5), (0, []))
        self.assertEqual(len(calls), 3)
        dead = []

        def always(url):
            dead.append(url)
            raise ConnectionResetError("reset by peer")

        with self.assertRaises(RuntimeError):
            EUtils(opener=always, sleep=lambda s: None).search("x", "2020/01/01", 5)
        self.assertEqual(len(dead), 5)


    def test_the_json_requests_pause_like_the_xml_requests(self):
        pauses, calls = [], []

        def opener(url):
            calls.append(url)
            if len(calls) < 3:
                raise OSError("reset")
            return b'{"esearchresult": {"count": "0", "idlist": []}}'
        EUtils(None, opener=opener, sleep=pauses.append).search("x", "2020/01/01", 5)
        self.assertEqual(pauses, [0.34, 1, 0.34, 2, 0.34])                         # the rate limit before each attempt, then 2 ** attempt after a failure


class EfetchTests(unittest.TestCase):
    XML = b"<PubmedArticleSet><PubmedArticle/></PubmedArticleSet>"

    def _eu(self, opener, key=None):
        return EUtils(key, opener=opener, sleep=lambda s: None)

    def test_the_url_names_the_records_the_tool_and_any_extra_parameter(self):
        urls = []

        def opener(url):
            urls.append(url)
            return self.XML
        self.assertEqual(self._eu(opener).efetch(["11", "12"], rettype="abstract"), self.XML)
        self.assertEqual(self._eu(opener, "KEY").efetch(["13"]), self.XML)
        self.assertEqual(urls[0], "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=11%2C12&retmode=xml&rettype=abstract&tool=thesis_research")
        self.assertEqual(urls[1], "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=13&retmode=xml&tool=thesis_research&api_key=KEY")

    def test_a_cut_connection_or_a_truncated_download_is_retried(self):
        calls = []

        def opener(url):
            calls.append(url)
            if len(calls) == 1:
                raise http.client.IncompleteRead(b"partial")
            if len(calls) == 2:
                return b"<PubmedArticleSet><PubmedArticle>"               # not well-formed: the download stopped half way
            if len(calls) == 3:
                raise ConnectionResetError("reset by peer")
            return self.XML
        self.assertEqual(self._eu(opener).efetch(["1"]), self.XML)
        self.assertEqual(len(calls), 4)

    def test_five_failed_attempts_are_an_error_that_names_the_cause(self):
        calls = []

        def opener(url):
            calls.append(url)
            raise TimeoutError("simulated")
        with self.assertRaises(RuntimeError) as caught:
            self._eu(opener).efetch(["1"])
        self.assertEqual(len(calls), 5)
        self.assertIn("simulated", str(caught.exception))

    def test_the_pause_before_a_request_follows_the_rate_limit_and_grows_after_a_failure(self):
        pauses = []
        calls = []

        def opener(url):
            calls.append(url)
            if len(calls) < 3:
                raise OSError("reset")
            return self.XML
        EUtils(None, opener=opener, sleep=pauses.append).efetch(["1"])
        self.assertEqual(pauses, [0.34, 1, 0.34, 2, 0.34])                         # delay before each attempt, then 2 ** attempt after a failure
        self.assertEqual(EUtils("KEY").delay, 0.11)


if __name__ == "__main__":
    unittest.main()
