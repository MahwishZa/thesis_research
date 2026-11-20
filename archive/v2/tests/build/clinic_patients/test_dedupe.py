"""Step dedupe (src/build/clinic_patients/dedupe.py) on the invented cohort of project.py."""

import io
from unittest import mock

import numpy as np
from PIL import Image

from src.build.clinic_patients import clean, dedupe, inventory
from tests.build.clinic_patients.project import Cohort


def edited(content: bytes, pixels: int) -> bytes:
    """The same picture with its first ``pixels`` pixels made black: a near copy."""
    picture = Image.open(io.BytesIO(content)).convert("RGB")
    for x in range(pixels):
        picture.putpixel((x % picture.width, x // picture.width), (0, 0, 0))
    out = io.BytesIO()
    picture.save(out, format="PNG")
    return out.getvalue()


def bits(a: np.ndarray, b: np.ndarray) -> int:
    return int(dedupe.POPCOUNT[a ^ b].sum())


class DedupeTests(Cohort):
    def prepare(self):
        inventory.inventory(self.data)
        clean.clean(self.data)

    def test_the_hash_of_a_picture_and_the_distance_of_two_by_hand(self):
        rising = Image.new("L", (17, 16))
        rising.putdata([16 * x for _ in range(16) for x in range(17)])
        falling = rising.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for name, picture in (("rising.png", rising), ("falling.png", falling)):
            picture.save(self.base / name)
        # each bit says whether the right pixel is brighter than the left one: a rising picture has all 256 bits set, a falling one none
        self.assertEqual(dedupe.dhash(self.base / "rising.png").tolist(), [255] * 32)
        self.assertEqual(dedupe.dhash(self.base / "falling.png").tolist(), [0] * 32)
        hashes = np.array([[0] * 32, [255] * 32, [15] + [0] * 31], dtype=np.uint8)
        self.assertEqual(dedupe.distances(hashes, 0, 3).tolist(), [[0, 256, 4], [256, 0, 252], [4, 252, 0]])
        self.assertEqual(dedupe.distances(hashes, 1, 3).tolist(), [[256, 0, 252], [4, 252, 0]])

    def test_a_copy_is_dropped_a_picture_with_two_diagnoses_is_dropped_whole_and_an_unreadable_photo_is_dropped(self):
        self.prepare()
        done = dedupe.dedupe(self.data)
        self.assertEqual((done["clusters"], done["kept"], done["dropped"]),
                         (2, 121, ["P900_1_b.png", "P901_1_a.png", "P901_2_a.png", "P902_1_a.png"]))
        rows = {r["img_id"]: r for r in self.table("duplicates.csv")}
        self.assertEqual(list(rows), ["P900_1_a.png", "P900_1_b.png", "P901_1_a.png", "P901_2_a.png", "P902_1_a.png"])
        self.assertEqual([(r["cluster"], r["decision"], r["reason"]) for r in rows.values()],
                         [("C001", "keep", "kept_one_of_cluster"), ("C001", "drop", "identical_copy"),
                          ("C002", "drop", "conflicting_labels"), ("C002", "drop", "conflicting_labels"), ("", "drop", "unreadable")])
        self.assertEqual((rows["P900_1_b.png"]["bits_from_kept"], rows["P901_1_a.png"]["bits_from_kept"]), ("0", ""))
        self.assertEqual(len(self.table("photo_hashes.csv")), 124)  # every photo that can be opened
        self.assertEqual({r["img_id"] for r in self.table("photo_hashes.csv")} & {"P902_1_a.png"}, set())

    def test_every_pair_up_to_the_link_distance_is_listed_with_who_they_belong_to(self):
        self.prepare()
        dedupe.dedupe(self.data)
        pairs = self.table("near_pairs.csv")
        self.assertEqual([(p["img_a"], p["img_b"], p["bits"], p["same_patient"], p["same_lesion"], p["label_a"], p["label_b"], p["same_cluster"])
                          for p in pairs],
                         [("P900_1_a.png", "P900_1_b.png", "0", "yes", "yes", "BCC", "BCC", "yes"),
                          ("P901_1_a.png", "P901_2_a.png", "0", "yes", "no", "NEV", "ACK", "yes")])

    def test_the_report_shows_the_rule_the_distances_and_the_decisions_by_class(self):
        self.prepare()
        dedupe.dedupe(self.data)
        report = self.report()
        for expected in ("At most **8** bits apart (identical files are 0 bits apart): one picture.",
                         "| 0 | 4 |", "| 1 to 8 | 0 |", "| 9 to 16 | 0 |", "| 65 to 256 | 120 |",  # nearest other photo of 124 photos
                         "- Clusters with conflicting diagnoses (all photos dropped): 1",
                         "- Clusters with one diagnosis (one photo stays): 1",
                         "| BCC | 22 | 1 | 0 | 1 |", "| NEV | 21 | 1 | 1 | 0 |", "| ACK | 21 | 1 | 1 | 0 |", "| SEK | 21 | 1 | 0 | 0 |",
                         "- C001: `P900_1_a.png` (BCC), `P900_1_b.png` (BCC) (`P900_1_a.png` stays)",
                         "- C002: `P901_1_a.png` (NEV), `P901_2_a.png` (ACK) (conflicting: all dropped)"):
            self.assertIn(expected, report)

    def _near_copies(self):
        """P903 (MEL): a near copy of P200's photo, same diagnosis. P904 (MEL): a near copy of a nevus photo, so another diagnosis."""
        self.add("P903", "1", "MEL", "P903_1_a.png", edited(self.photos["P200_1_a.png"], 25), {"age": "40", "region": "ARM"})
        self.add("P904", "1", "MEL", "P904_1_a.png", edited(self.photos["P300_1_a.png"], 25), {"age": "41", "region": "ARM"})
        self.write()
        self.prepare()
        h = {i: dedupe.dhash(self.raw / "images" / "part_1" / i) for i in ("P200_1_a.png", "P903_1_a.png", "P300_1_a.png", "P904_1_a.png")}
        return bits(h["P200_1_a.png"], h["P903_1_a.png"]), bits(h["P300_1_a.png"], h["P904_1_a.png"])

    def test_near_copies_under_the_duplicate_distance_are_one_picture(self):
        d1, d2 = self._near_copies()
        self.assertTrue(0 < d1 < 60 and 0 < d2 < 60, (d1, d2))
        with mock.patch.object(dedupe, "DUP_BITS", max(d1, d2)), mock.patch.object(dedupe, "LINK_BITS", max(d1, d2) + 4):
            done = dedupe.dedupe(self.data)
        rows = {r["img_id"]: r for r in self.table("duplicates.csv")}
        self.assertEqual((rows["P200_1_a.png"]["decision"], rows["P903_1_a.png"]["decision"], rows["P903_1_a.png"]["reason"]),
                         ("keep", "drop", "near_copy"))  # same diagnosis: the smaller img_id stays
        self.assertEqual((rows["P300_1_a.png"]["decision"], rows["P904_1_a.png"]["decision"], rows["P904_1_a.png"]["reason"]),
                         ("drop", "drop", "conflicting_labels"))  # one picture, two diagnoses
        self.assertEqual(rows["P903_1_a.png"]["bits_from_kept"], str(d1))
        self.assertEqual(done["clusters"], 4)

    def test_near_copies_between_the_duplicate_and_the_link_distance_stay_but_are_listed_for_the_split(self):
        d1, d2 = self._near_copies()
        with mock.patch.object(dedupe, "DUP_BITS", min(d1, d2) - 1), mock.patch.object(dedupe, "LINK_BITS", max(d1, d2)):
            done = dedupe.dedupe(self.data)
        self.assertEqual(done["clusters"], 2)  # only the exact copies of the cohort
        pairs = [p for p in self.table("near_pairs.csv") if p["same_patient"] == "no"]
        self.assertEqual([(p["img_a"], p["img_b"], p["same_cluster"]) for p in pairs],
                         sorted([("P200_1_a.png", "P903_1_a.png", "no"), ("P300_1_a.png", "P904_1_a.png", "no")]))
        self.assertNotIn("P903_1_a.png", done["dropped"])
        self.assertIn("- Pairs of photos of **different patients** at most", self.report())

    def test_the_steps_before_it_must_have_run_and_agree(self):
        with self.assertRaisesRegex(FileNotFoundError, "run the clean step first"):
            dedupe.dedupe(self.data)
        self.prepare()
        self.add("P905", "1", "ACK", "P905_1_a.png", b"x", {"age": "50", "region": "ARM"})
        self.write()
        clean.clean(self.data)  # a photo of the cleaned table that the inventory has not seen
        with self.assertRaisesRegex(ValueError, "not in build/raw_images.csv"):
            dedupe.dedupe(self.data)

    def test_the_components_helper_joins_chains_in_a_fixed_order(self):
        from src.build.clinic_patients.table import components
        self.assertEqual(components(["d", "c", "b", "a", "e"], [("d", "c"), ("c", "b")]), [["a"], ["b", "c", "d"], ["e"]])
        self.assertEqual(components(["a", "b"], []), [["a"], ["b"]])
        self.assertEqual(components(["b", "a", "c"], [("c", "a")]), components(["a", "b", "c"], [("a", "c"), ("c", "a")]))


if __name__ == "__main__":
    import unittest
    unittest.main()
