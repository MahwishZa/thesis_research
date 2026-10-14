"""Which questions of MedRevQA no split uses yet, and which of them name dementia or Alzheimer's disease.

A question is *used* when its study group, its Cochrane ID or its wording appears in a split of ``data/benchmark.jsonl``. The
Alzheimer's disease set (``ad_benchmark``) and the fresh questions (``fresh_supply``, ``fresh_benchmark``) apply the same
exclusion rules to the rest of MedRevQA; this module holds those rules so that the three commands cannot drift apart.
"""

from __future__ import annotations

import re

from .benchmark import read_version

#: The wording of a question of the Alzheimer's disease set (and the wording the fresh questions must not have).
AD_QUESTION = re.compile(r"dementia|alzheimer|mild cognitive impairment|cognitive decline|cognitively impaired|"
                         r"cognitive impairment", re.I)
#: Review text (question, objectives, conclusions) that names Alzheimer's disease or dementia. Counted because a review can be about
#: dementia while its question does not say so.
AD_DOMAIN_TEXT = re.compile(r"alzheimer|dementia", re.IGNORECASE)
#: Neurodegenerative diseases other than dementia itself (the dementia wording is excluded from the fresh pool already). Counted for a possible
#: descriptive stratum "neurodegenerative diseases related to Alzheimer's disease"; it is not Alzheimer's disease and is never called that.
NEURO_TEXT = re.compile(r"parkinson|lewy bod|frontotemporal|huntington|amyotrophic|motor neuron|neurodegenerat|supranuclear|multiple system atrophy|"
                        r"corticobasal|prion|creutzfeldt", re.IGNORECASE)


def used_review_ids(main_rows: list[dict]) -> frozenset:
    """Cochrane IDs of every review (either version) that an item of the main benchmark is built from."""
    return frozenset(v["cochrane_id"] for r in main_rows for v in (r["newest"], r["previous"])
                     if v.get("cochrane_id"))


def candidates(medrev: dict, groups: dict, used_groups: set) -> list[tuple[int, int, str, object]]:
    """(newest row, previous row, kind, group id or None) for every group or ungrouped row outside ``used_groups``."""
    grouped = {k for keys in groups.values() for k in keys}
    out = []
    for gid in sorted(groups):
        if gid in used_groups:
            continue
        keys = sorted(set(groups[gid]))
        if len(keys) >= 2:
            same = medrev[keys[0]]["Label"].strip() == medrev[keys[1]]["Label"].strip()
            out.append((keys[0], keys[1], "unchanged" if same else "changed", gid))
        else:
            out.append((keys[0], keys[0], "unchanged", gid))
    out += [(k, k, "unchanged", None) for k in sorted(medrev) if k not in grouped]
    return sorted(out, key=lambda t: t[:3])


def fresh_items(medrev: dict, groups: dict, used_groups: set, used_reviews: frozenset, used_questions: frozenset) -> list[dict]:
    seen, out = set(), []
    for newest, previous, kind, gid in candidates(medrev, groups, used_groups):
        q = medrev[newest]["Question"].strip()
        key = q.lower()
        try:
            nv, pv = read_version(medrev, newest), read_version(medrev, previous)
        except Exception:                                  # a citation without a date cannot be asked as of a date
            continue
        if key in seen or key in used_questions or {nv.cochrane_id, pv.cochrane_id} & used_reviews:
            continue
        seen.add(key)
        out.append({"row": newest, "previous": previous, "group_id": gid, "kind": kind, "single_version": newest == previous, "date": nv.date,
                    "precision": nv.date_precision, "label": nv.label, "cochrane_id": nv.cochrane_id,
                    "dementia_wording": bool(AD_QUESTION.search(q)),
                    "neuro_wording": bool(NEURO_TEXT.search(q)),
                    "neuro_in_review_text": bool(NEURO_TEXT.search(" ".join(
                        [q, medrev[newest].get("objectives") or "", medrev[newest].get("conclusions") or ""]))),
                    "dementia_in_review_text": bool(AD_DOMAIN_TEXT.search(" ".join(
                        [q, medrev[newest].get("objectives") or "", medrev[newest].get("conclusions") or ""])))})
    return out
