"""Fixed seeds and file fingerprints (SHA-256), so a run can be repeated and its inputs checked."""
import hashlib

SEED = 42


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# The fingerprint of a text file (questions, predictions) is taken with LF line endings, so the same file has the same fingerprint
# on Windows (CRLF when an older version of the code wrote it) and in Git (always LF, .gitattributes). Before 2026-11-25 the reports
# recorded the fingerprint of the file as Windows saved it (CRLF); text_matches accepts that form too, so an approved report stays valid.
def _lf_bytes(path):
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def text_sha256(path):
    """SHA-256 of a text file with its line endings written as LF."""
    return hashlib.sha256(_lf_bytes(path)).hexdigest()


def text_matches(path, recorded):
    """Does a recorded fingerprint belong to this text file, with LF (now) or CRLF (Windows, reports before 2026-11-25) endings?"""
    lf = _lf_bytes(path)
    return recorded in (hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest())
