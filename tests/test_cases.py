"""The case datasets ("cases": the case text and its published photos; "case_photos": the photos only), on a small invented
project laid out like the repository: no network, no real data."""

import hashlib
import io
import json
import shutil
from unittest import mock

from PIL import Image

from src import evaluate
from tests.test_evaluate import TEXT_CONFIG, FakeModel, TextProject

CASES = ["train/actinic keratosis/101-1.txt", "train/melanoma/102-2.txt", "train/melanoma/103-3.txt", "train/nevus/104-4.txt",
         "val/melanoma/107-7.txt", "test/actinic keratosis/201-9.txt", "test/melanoma/203-11.txt", "test/melanoma/204-13.txt",
         "test/nevus/205-14.txt", "test/nevus/207-16.txt"]  # 10 of the text project's 16 cases have photos


def picture(colour) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (20, 10), colour).save(out, format="PNG")
    return out.getvalue()


class Recorder(FakeModel):
    def __init__(self):
        super().__init__()
        self.requests = []

    def predict(self, images, prompt):
        self.requests.append((images, prompt))
        return super().predict(images, prompt)


class CaseProject(TextProject):
    def setUp(self):
        super().setUp()
        rows, self.pictures = [], {}
        for number, path in enumerate(CASES):
            split, label, name = path.split("/")
            files = []
            for k in range(1 + number % 3):  # one, two or three photos
                content = picture((10 * number, 20 * k, 7))
                file = f"figures/images/{name[:-4]}/f{k + 1}.png"
                (self.base / "data" / "multimodal" / file).parent.mkdir(parents=True, exist_ok=True)
                (self.base / "data" / "multimodal" / file).write_bytes(content)
                files.append((file, hashlib.sha256(content).hexdigest()))
            self.pictures[path] = files
            sha = hashlib.sha256(self.texts[path].encode("utf-8")).hexdigest()
            rows.append(f"{split},{label},{path},{sha},{name.split('-')[0]},{';'.join(f'{f}|{s}' for f, s in files)},{len(files)}\n")
        self.cases_table = self.base / "data" / "multimodal" / "cases.csv"
        self.cases_table.write_text("split,label,path,sha256,pmid,pictures,n_pictures\n" + "".join(rows), encoding="utf-8")
        self.photo_prompt = self.base / "photos.txt"
        self.photo_prompt.write_text("Photos of a lesion.\nPick one:\n{labels}\nAnswer with the name only.\n", encoding="utf-8")

    def config_for(self, dataset):
        prompt = self.prompt if dataset == "cases" else self.photo_prompt
        path = self.base / f"{dataset}.toml"
        path.write_text(TEXT_CONFIG.format(prompt=prompt.as_posix()).replace('dataset = "text"', f'dataset = "{dataset}"'), encoding="utf-8")
        return path

    def ask(self, dataset, model, **kwargs):
        with mock.patch("src.evaluate.load", return_value=model):
            return evaluate.run(str(self.config_for(dataset)), "test", results=self.results, **kwargs)


class CaseDatasetTests(CaseProject):
    def test_the_cases_load_as_a_subset_of_the_text_dataset_with_their_photos(self):
        for dataset in ("cases", "case_photos"):
            items = evaluate.load_dataset(*evaluate.dataset_paths(dataset), dataset)  # the text dataset's other cases are expected
            self.assertEqual(sorted(i.path for i in items), sorted(CASES))
            self.assertEqual({i.path: list(i.pictures) for i in items}, self.pictures)
            self.assertEqual(items[0].group, "101")

    def test_the_questions_fingerprint_changes_with_any_photo(self):
        items = evaluate.read_table(self.cases_table, "cases")
        moved = [items[0].__class__(**{**items[0].__dict__, "pictures": items[0].pictures[::-1] + (("x.png", "f" * 64),)}), *items[1:]]
        self.assertNotEqual(evaluate.items_hash(items), evaluate.items_hash(moved))
        self.assertEqual(evaluate.items_hash(items), evaluate.items_hash(list(reversed(items))))

    def test_a_changed_or_missing_photo_stops_the_run(self):
        file = self.base / "data" / "multimodal" / self.pictures[CASES[6]][0][0]
        file.write_bytes(picture((1, 2, 3)))
        with self.assertRaisesRegex(ValueError, "1 changed, 0 missing"):
            evaluate.load_dataset(*evaluate.dataset_paths("cases"), "cases")
        file.unlink()
        with self.assertRaisesRegex(ValueError, "0 changed, 1 missing"):
            evaluate.load_dataset(*evaluate.dataset_paths("cases"), "cases")

    def test_the_table_must_give_each_case_its_photos_once(self):
        text = self.cases_table.read_text(encoding="utf-8")
        lines = text.splitlines()
        first = lines[1].split(",")
        self.cases_table.write_text("\n".join([lines[0], ",".join(first[:5] + ["", "0"]), *lines[2:]]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lists no picture"):
            evaluate.read_table(self.cases_table, "cases")
        second = lines[2].split(",")
        self.cases_table.write_text("\n".join([lines[0], lines[1], ",".join(second[:5] + [first[5], "1"]), *lines[3:]]) + "\n",
                                    encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "the same picture is also on line 2"):
            evaluate.read_table(self.cases_table, "cases")
        self.cases_table.write_text(text.replace("|", "#", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file\\|sha256"):
            evaluate.read_table(self.cases_table, "cases")

    def test_the_text_and_photos_run_sends_every_photo_in_order_and_the_case_in_the_prompt(self):
        model = Recorder()
        out = self.ask("cases", model, run_id="t1")
        preds = {p["path"]: p for p in (json.loads(x) for x in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines())}
        self.assertEqual(len(model.requests), 5)
        for images, prompt in model.requests:
            path = next(p for p in preds if self.texts[p] in prompt)  # the case text, word for word
            expected = [(self.base / "data" / "multimodal" / f).read_bytes() for f, _ in self.pictures[path]]
            self.assertEqual([content for content, _ in images], expected)
            self.assertEqual({mime for _, mime in images}, {"image/png"})
            sent = preds[path]["case_sent"]
            self.assertEqual([s["sha256"] for s in sent["pictures"]], [s for _, s in self.pictures[path]])
            self.assertEqual(sent["case"]["sha256"], hashlib.sha256(self.texts[path].encode("utf-8")).hexdigest())

    def test_the_photos_run_sends_the_photos_and_no_case_text(self):
        model = Recorder()
        self.ask("case_photos", model, run_id="p1")
        self.assertEqual(len(model.requests), 5)
        for images, prompt in model.requests:
            self.assertTrue(images)
            self.assertFalse(any(text in prompt for text in self.texts.values()))
            self.assertNotIn("Case:", prompt)
        self.photo_prompt.write_text("Photos.\nCase: {case}\n{labels}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must not contain"):
            self.ask("case_photos", Recorder(), run_id="p2")

    def test_a_text_run_is_compared_with_a_case_run_on_the_shared_cases_only(self):
        with mock.patch("src.evaluate.load", return_value=FakeModel()):
            text_run = evaluate.run(str(self.config), "test", run_id="t", results=self.results)
        cases_run = self.ask("cases", FakeModel(), run_id="c")
        photos_run = self.ask("case_photos", FakeModel(), run_id="p")
        for a, b in ((text_run, cases_run), (cases_run, text_run), (photos_run, cases_run)):
            found = evaluate.compare(a, b)
            self.assertEqual(found["n"], 5)
        other = self.results / "image-copy"
        shutil.copytree(cases_run, other)
        card = json.loads((other / "run_card.json").read_text(encoding="utf-8"))
        card["config"]["dataset"], card["n_images"] = "multimodal", card["n_cases"]  # photos of other patients: nothing pairs
        (other / "run_card.json").write_text(json.dumps(card), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differ in their dataset"):
            evaluate.compare(text_run, other)

    def test_the_setup_check_sends_two_blank_pictures_for_a_case_dataset(self):
        model = Recorder()
        answer, problem = evaluate._probe(model, print, images=2)
        self.assertIsNone(problem)
        ((images, prompt),) = model.requests
        self.assertEqual((len(images), prompt), (2, evaluate.PROBE_PROMPT))
