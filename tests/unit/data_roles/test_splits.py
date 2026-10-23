"""data_roles.splits beyond the acceptance tests: the exact algorithm, the order of the role names, and the error messages."""

import hashlib
import unittest

from data_roles import splits


def expected_point(question_id, seed):
    """The documented algorithm, written again here: the first 8 bytes of SHA-256(seed NUL id) as a fraction of 2**64."""
    digest = hashlib.sha256(f"{seed}\x00{question_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class AlgorithmTests(unittest.TestCase):
    def test_a_split_in_halves_equals_the_documented_hash_rule(self):
        ids = [f"q{i}" for i in range(200)]
        roles = splits.split_by_question(question_ids=ids, fractions={"a": 0.5, "b": 0.5}, seed=3)
        for question_id in ids:
            with self.subTest(question_id=question_id):
                self.assertEqual(roles[question_id], "a" if expected_point(question_id, 3) < 0.5 else "b")

    def test_the_roles_of_known_ids_never_change(self):
        # A pinned result: a change of the hash, the seed format or the cut points would move questions between roles.
        ids = [f"q{i}" for i in range(16)]
        roles = splits.split_by_question(question_ids=ids, fractions={"a": 0.5, "b": 0.5}, seed=7)
        self.assertEqual("".join(roles[i] for i in ids), "aaaaaaaabbabbbbb")
        roles = splits.split_by_question(question_ids=ids, fractions={"a": 0.5, "b": 0.5}, seed=8)
        self.assertEqual("".join(roles[i] for i in ids), "aabbbbbabbbabbbb")

    def test_the_order_in_which_the_roles_are_listed_does_not_matter(self):
        ids = [f"q{i}" for i in range(500)]
        one = splits.split_by_question(question_ids=ids, fractions={"train": 0.8, "validation": 0.1, "test": 0.1}, seed=7)
        two = splits.split_by_question(question_ids=ids, fractions={"test": 0.1, "validation": 0.1, "train": 0.8}, seed=7)
        self.assertEqual(one, two)

    def test_a_role_with_a_zero_share_gets_no_question(self):
        roles = splits.split_by_question(question_ids=[f"q{i}" for i in range(300)], fractions={"fit": 1.0, "none": 0.0}, seed=1)
        self.assertEqual(set(roles.values()), {"fit"})

    def test_the_shares_may_be_sums_of_floats(self):
        # 0.1 + 0.2 + 0.7 is 0.9999999999999999 in floating point; that is a valid set of shares.
        roles = splits.split_by_question(question_ids=["a", "b"], fractions={"x": 0.1, "y": 0.2, "z": 0.7}, seed=0)
        self.assertEqual(set(roles), {"a", "b"})

    def test_an_id_that_is_not_text_is_an_error(self):
        for bad in (5, None, b"q1"):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                splits.split_by_question(question_ids=[bad], fractions={"a": 1.0}, seed=0)

    def test_bad_shares_name_the_problem(self):
        for fractions in ({"a": float("nan"), "b": 1.0}, {"a": True, "b": 0.0}, {"": 1.0}, {"a": "0.5", "b": 0.5}):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                splits.split_by_question(question_ids=["q"], fractions=fractions, seed=0)

    def test_the_generator_of_ids_is_read_once(self):
        roles = splits.split_by_question(question_ids=(f"q{i}" for i in range(50)), fractions={"a": 1.0}, seed=0)
        self.assertEqual(len(roles), 50)


class DisjointTests(unittest.TestCase):
    def test_the_message_names_the_roles_that_share_an_id(self):
        with self.assertRaises(ValueError) as caught:
            splits.assert_disjoint({"train": ["x"], "test": ["x"]})
        self.assertIn("x in test, train", str(caught.exception))

    def test_many_shared_ids_are_counted_and_only_some_are_shown(self):
        shared = [f"id{i:02d}" for i in range(25)]
        with self.assertRaises(ValueError) as caught:
            splits.assert_disjoint({"train": shared, "test": shared})
        message = str(caught.exception)
        self.assertIn("25 in all", message)
        self.assertIn("id00", message)
        self.assertNotIn("id24", message)

    def test_an_id_in_three_roles_is_named_once_with_all_three(self):
        with self.assertRaises(ValueError) as caught:
            splits.assert_disjoint({"a": ["x"], "b": ["x"], "c": ["x"]})
        self.assertIn("x in a, b, c", str(caught.exception))

    def test_the_roles_of_a_split_are_always_disjoint(self):
        roles = splits.split_by_question(question_ids=[f"q{i}" for i in range(1000)], fractions={"a": 0.3, "b": 0.7}, seed=4)
        grouped = {"a": [], "b": []}
        for question_id, role in roles.items():
            grouped[role].append(question_id)
        self.assertIsNone(splits.assert_disjoint(grouped))


if __name__ == "__main__":
    unittest.main()
