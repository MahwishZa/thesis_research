"""Acceptance tests for docs/methodology.md section 7.7 (data roles). They skip until src/data_roles exists."""

import os
import random
import subprocess
import sys

from tests import SRC
from tests.spec.spec_targets import TARGETS, SpecTestCase, resolve

FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}
IDS = [f"q{i:05d}" for i in range(20000)]


def split(ids, fractions=FRACTIONS, seed=7):
    return resolve("split_by_question")(question_ids=ids, fractions=fractions, seed=seed)


class SplitTests(SpecTestCase):
    """The role of a question depends on its id and the seed only: not on row order, process or the other questions."""

    requires = ("split_by_question",)

    def test_every_question_gets_exactly_one_role_from_the_fractions(self):
        roles = split(IDS)
        self.assertEqual(set(roles), set(IDS))
        self.assertLessEqual(set(roles.values()), set(FRACTIONS))

    def test_the_shares_follow_the_fractions(self):
        roles = split(IDS)
        for role, fraction in FRACTIONS.items():
            with self.subTest(role=role):
                share = sum(1 for assigned in roles.values() if assigned == role) / len(IDS)
                self.assertAlmostEqual(share, fraction, delta=0.01)

    def test_the_same_inputs_give_the_same_split(self):
        self.assertEqual(split(IDS), split(IDS))

    def test_the_order_of_the_rows_does_not_matter(self):
        shuffled = list(IDS)
        random.Random(2).shuffle(shuffled)
        self.assertEqual(split(shuffled), split(IDS))

    def test_a_repeated_id_is_one_question(self):
        self.assertEqual(split(IDS + IDS[:100]), split(IDS))

    def test_a_question_keeps_its_role_when_others_are_added_or_removed(self):
        full = split(IDS)
        for question_id, role in split(IDS[::3]).items():
            self.assertEqual(full[question_id], role)
        extended = split(IDS + [f"new{i}" for i in range(500)])
        for question_id in IDS:
            self.assertEqual(extended[question_id], full[question_id])

    def test_another_seed_gives_another_split(self):
        first, second = split(IDS, seed=1), split(IDS, seed=2)
        moved = sum(1 for question_id in IDS if first[question_id] != second[question_id])
        self.assertGreater(moved / len(IDS), 0.1)

    def test_other_role_names_and_fractions_work(self):
        roles = split(IDS, fractions={"fit": 0.5, "hold": 0.5})
        self.assertEqual(set(roles.values()), {"fit", "hold"})
        self.assertAlmostEqual(sum(1 for role in roles.values() if role == "fit") / len(IDS), 0.5, delta=0.02)

    def test_no_questions_give_an_empty_split(self):
        self.assertEqual(split([]), {})

    def test_bad_fractions_are_errors(self):
        for bad in ({}, {"train": 0.5, "test": 0.4}, {"train": 1.2, "test": -0.2}, {"train": 0.6, "test": 0.6}):
            with self.subTest(fractions=bad), self.assertRaises(ValueError):
                split(IDS[:10], fractions=bad)

    def test_the_split_does_not_depend_on_the_python_process(self):
        # Python's built-in hash() of a string changes from one process to the next; a split built on it would too.
        module, name = TARGETS["split_by_question"]
        resolve("split_by_question")
        code = (
            "import importlib, json, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "split = getattr(importlib.import_module(sys.argv[2]), sys.argv[3])\n"
            "ids = ['q' + str(i) for i in range(500)]\n"
            "roles = split(question_ids=ids, fractions={'train': 0.8, 'validation': 0.1, 'test': 0.1}, seed=7)\n"
            "print(json.dumps(roles, sort_keys=True))\n"
        )
        outputs = []
        for hash_seed in ("1", "2", "3"):
            done = subprocess.run(
                [sys.executable, "-c", code, str(SRC), module, name],
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            outputs.append(done.stdout)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])


class DisjointTests(SpecTestCase):
    requires = ("assert_disjoint",)

    @staticmethod
    def check(roles):
        return resolve("assert_disjoint")(roles)

    def test_disjoint_roles_pass(self):
        self.assertIsNone(self.check({"train": ["a", "b"], "validation": ["c"], "test": ["d"]}))

    def test_repeated_ids_inside_one_role_are_fine(self):
        self.assertIsNone(self.check({"train": ["a", "a", "b"], "test": ["c", "c"]}))

    def test_sets_and_tuples_work(self):
        self.assertIsNone(self.check({"train": {"a", "b"}, "validation": ("c",), "test": frozenset({"d"})}))

    def test_empty_roles_pass(self):
        self.assertIsNone(self.check({}))
        self.assertIsNone(self.check({"train": [], "test": []}))

    def test_a_shared_id_is_an_error_that_names_it(self):
        with self.assertRaises(ValueError) as caught:
            self.check({"train": ["a", "q-shared-17"], "test": ["q-shared-17", "c"]})
        self.assertIn("q-shared-17", str(caught.exception))

    def test_every_pair_of_roles_is_checked(self):
        names = ("train", "validation", "test")
        for first in range(3):
            for second in range(first + 1, 3):
                roles = {name: ["own-" + name] for name in names}
                roles[names[first]].append("shared")
                roles[names[second]].append("shared")
                with self.subTest(pair=(names[first], names[second])), self.assertRaises(ValueError):
                    self.check(roles)
