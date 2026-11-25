"""Step 3 of the knowledge corpus: cutting a document's paragraphs into passages."""
import re

from src.knowledge.extract import clean


def sentences(text):
    """Splits at a full stop, question or exclamation mark followed by a space and a capital letter, digit or bracket."""
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", clean(text)) if s]


def chunk(text, max_words=200, overlap=1):
    """Passages of whole sentences, at most max_words words (a longer single sentence is cut), with `overlap` sentences shared."""
    out, cur = [], []
    for s in sentences(text):
        words = s.split()
        while len(words) > max_words:                       # a very long "sentence" (e.g. a list): cut it
            if cur:
                out.append(" ".join(cur))
                cur = []
            out.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if cur and len(" ".join(cur).split()) + len(words) > max_words:
            out.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            if len(" ".join(cur).split()) + len(words) > max_words:
                cur = []
        cur.append(" ".join(words))
    if cur and (not out or " ".join(cur) not in out[-1]):
        out.append(" ".join(cur))
    return [p for p in out if p]
