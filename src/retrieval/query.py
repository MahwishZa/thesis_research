"""The retrieval query of each view, made only of the question's own words (never the options, the true class or a tag)."""
import os
import re

GENERIC_QUERY = "dermoscopic image of a skin lesion"      # the query of an image-only question: it has no patient facts
GENERIC_TEXT_QUERY = "skin lesion"                      # the query of a text or image + text question with no recorded fact


def patient_facts(q):
    """The patient sentence of a question without the generic question and the words about the image: "" for an image-only question."""
    text = q["question"]
    for phrase in ("What is the most likely diagnosis?", "Look at the dermoscopic image of the skin lesion.",
                   "Look at the dermoscopic image of the lesion."):
        text = text.replace(phrase, "")
    return " ".join(text.split())


def recorded_facts(text):
    """The patient sentence without what is "not recorded": "Patient: age not recorded, female. Skin lesion on the trunk." gives
    "Patient: female. Skin lesion on the trunk."; "" when nothing is recorded."""
    text = re.sub(r"\s*Lesion site not recorded\.?", "", text)            # an unknown site
    text = re.sub(r"\bage not recorded,?\s*", "", text)                   # an unknown age
    text = re.sub(r",?\s*\bsex not recorded", "", text)                   # an unknown sex
    text = " ".join(text.split())
    text = re.sub(r"^Patient:\s*\.\s*", "", text)                        # nothing left of the patient sentence: "Patient: ."
    if text == "Patient:":
        return ""
    return text if text.endswith((".", ")")) or not text else text + "."


def retrieval_query(q, view="text_only"):
    """The text the retriever gets: the recorded patient facts of the question. When there are none, a fixed sentence that fits the
    view (the image-only view has no facts at all, and a text view has no image). Never the options, the true class or a tag."""
    facts = recorded_facts(patient_facts(q))
    if facts:
        return facts
    return GENERIC_QUERY if view == "image_only" else GENERIC_TEXT_QUERY


def image_query_vector(q, vectors):
    name = os.path.basename(q["image_path"])
    if name not in vectors:
        raise KeyError(f"no saved image vector for {name}: run python -m src.main visual-index again")
    return vectors[name]
