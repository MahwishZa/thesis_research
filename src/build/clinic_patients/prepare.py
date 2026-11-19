"""PAD-UFES-20, the clinic patients cohort of Version 2: smartphone photos of skin lesions, each with the clinical record of the same
patient visit (Pacheco et al., Data in Brief, 2020; Mendeley Data, doi 10.17632/zr7vgbcyr2.1). One command per step, in this order:

    python -m src.build.clinic_patients.prepare inventory    what the download holds (changes nothing)
    python -m src.build.clinic_patients.prepare clean        normalise, check and flag the clinical record
    python -m src.build.clinic_patients.prepare dedupe       find identical and near-identical photos, decide which stay
    python -m src.build.clinic_patients.prepare split        patient-level train, validation and test splits; dataset.csv
    python -m src.build.clinic_patients.prepare shortcuts    how much the gaps and the photo properties alone give away

or ``all`` for the five in order. The original download is never changed: it stays in data/clinic_patients/raw/ on this computer. Every
step writes its decisions to tables in build/ and its own section of dataset_report.md; the tables and the report are committed.
"""

from __future__ import annotations

import argparse
import sys

from . import clean, dedupe, inventory, shortcuts, split
from .table import STEPS

RUN = {"inventory": inventory.inventory, "clean": clean.clean, "dedupe": dedupe.dedupe, "split": split.split,
       "shortcuts": shortcuts.shortcuts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.build.clinic_patients.prepare", description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=[*STEPS, "all"], help="the step to run, or all of them in order")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for step in STEPS if args.step == "all" else [args.step]:
        try:
            done = RUN[step]()
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(f"ERROR in step {step}: {exc}") from exc
        print(f"[{step}] " + done["summary"])
    print("Written: dataset_report.md and the tables of build/ (and dataset.csv after split). Commit them all: git add data/clinic_patients")


if __name__ == "__main__":
    main(sys.argv[1:])
