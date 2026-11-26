"""The shared measures: statistics (intervals, McNemar, Holm, cosine), how an answer is read (true and predicted class,
coverage), and the hallucination measures (evidence check, image claims in text-only answers, semantic similarity)."""
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict

from src.utils import CLASSES, OPTION_TEXT


def percentile(sorted_values, q):
    """The q-th percentile with linear interpolation (the same as numpy's default)."""
    k = (len(sorted_values) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def bootstrap_ci(y, p, groups, fn, n=1000, seed=42):
    """95% percentile interval of fn. Whole lesions are drawn with replacement, so photos of one lesion stay together."""
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    keys, rng, values = list(members), random.Random(seed), []
    for _ in range(n):
        idx = [i for g in rng.choices(keys, k=len(keys)) for i in members[g]]
        values.append(fn([y[i] for i in idx], [p[i] for i in idx]))
    values.sort()
    return [round(percentile(values, 2.5), 4), round(percentile(values, 97.5), 4)]


def holm(p_values):
    """Holm's step-down correction: the i-th smallest of m p-values is multiplied by (m - i + 1), the results are made
    non-decreasing in that order and capped at 1. Returns the adjusted p-values in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def mcnemar_exact(y, pa, pb):
    """McNemar's exact test: a two-sided binomial test (p = 0.5) on the questions that exactly one system got right."""
    b = sum(a == t and c != t for t, a, c in zip(y, pa, pb))
    c = sum(a != t and x == t for t, a, x in zip(y, pa, pb))
    n = b + c
    p_value = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": round(p_value, 6)}


def pct(x, digits=1):
    """A fraction as a percentage for the screen (0.2542 -> "25.4%"); the files keep fractions."""
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def cosine(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


TEXT_TO_CLASS = {text: code for code, text in OPTION_TEXT.items()}
INVALID = "invalid"


def gold_class(q):
    """The gold class of a question; stops if the answer letter does not point to it (a broken question file)."""
    code = TEXT_TO_CLASS[q["options"][q["answer"]]]
    if code != q["gold_label"]:
        raise ValueError(f"{q['id']}: answer {q['answer']} is {code}, gold label is {q['gold_label']}")
    return code


def predicted_class(q, p):
    """The class a prediction chose, read with this case's options; 'invalid' when there is no usable option letter."""
    letter = str((p or {}).get("pred") or "").strip().upper()
    return TEXT_TO_CLASS[q["options"][letter]] if letter in q["options"] else INVALID


def coverage(y, answers):
    """Selective classification: share of all questions answered with a valid diagnosis marked "yes" (here: with a verified quote)."""
    return sum(c != INVALID and flag == "yes" for c, flag in answers) / len(y)


def selective_risk(y, answers):
    """Share of wrong diagnoses among the questions marked "yes" (None when none is)."""
    covered = [(t, c) for t, (c, flag) in zip(y, answers) if c != INVALID and flag == "yes"]
    return sum(t != c for t, c in covered) / len(covered) if covered else None


SIMILARITY_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"     # a biomedical sentence encoder (PubMedBERT tuned on MS MARCO)


def sentence_encoder(model=SIMILARITY_MODEL):
    """A function that turns a list of texts into a list of embedding vectors (downloads the model on first use)."""
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model)

    def encode(texts):
        return enc.encode(list(texts), convert_to_numpy=True).tolist()
    # the model reads at most max_seq_length tokens of a text and ignores the rest: counted so a report can say so
    encode.max_tokens = enc.max_seq_length
    encode.n_tokens = lambda text: len(enc.tokenizer(text)["input_ids"])
    return encode


def reference_lists(reference_texts):
    """{class: [text, ...]} from any of the accepted forms: {class: text}, {class: {"text": ...}} or the file of
    the references command, {class: {"texts": [{"text": ...}, ...]}}."""
    out = {}
    for c, r in reference_texts.items():
        if isinstance(r, str):
            out[c] = [r]
        elif "texts" in r:
            out[c] = [t["text"] for t in r["texts"]]
        else:
            out[c] = [r["text"]]
        if not out[c]:
            raise ValueError(f"the class {c} has no reference text")
    return out


def semantic_similarity(qa, pred, reference_texts, encode):
    """Cosine similarity between the embedding of each model explanation ("rationale") and the embeddings of the reference texts of
    the question's true class, keeping the highest (the best-matching reference, as in multi-reference scoring); then the mean
    over the questions, with a 95% interval over lesions and the mean per class.
    reference_texts: see reference_lists; the texts are quoted from a cited source, never written by us."""
    # Only answers with a usable letter: a refusal ("I'm sorry, I can't help") or a cut-off reply has no reasoning to compare
    # (corrected 2026-11-24: the first image runs scored GPT-4o's refusal sentences as reasoning).
    ids = [i for i in qa if (pred.get(i) or {}).get("rationale") and predicted_class(qa[i], pred[i]) != INVALID]
    if not ids:
        return None
    refs = reference_lists(reference_texts)
    missing = sorted({qa[i]["gold_label"] for i in ids} - set(refs))
    if missing:
        raise ValueError(f"no reference text for the classes {missing}")
    classes = sorted({qa[i]["gold_label"] for i in ids})
    flat = [(c, t) for c in classes for t in refs[c]]
    ref_vec = defaultdict(list)
    for (c, _), v in zip(flat, encode([t for _, t in flat])):
        ref_vec[c].append(v)
    rationales = [pred[i]["rationale"] for i in ids]
    sims = [max(cosine(h, r) for r in ref_vec[qa[i]["gold_label"]]) for i, h in zip(ids, encode(rationales))]
    groups = [qa[i].get("group_id", i) for i in ids]
    per_class = defaultdict(list)
    for i, v in zip(ids, sims):
        per_class[qa[i]["gold_label"]].append(v)
    out = {"mean": round(mean(sims), 4), "95ci": bootstrap_ci(sims, sims, groups, lambda a, b: mean(a)),
           "per_class": {c: round(mean(per_class[c]), 4) for c in CLASSES if c in per_class},
           "n_questions_with_rationale": len(ids), "n_reference_texts": {c: len(refs[c]) for c in classes},
           "aggregation": "highest cosine over the reference texts of the true class"}
    if hasattr(encode, "n_tokens"):
        out["longer_than_the_model_reads"] = {"max_tokens": encode.max_tokens,
                                              "reference_texts": sum(encode.n_tokens(t) > encode.max_tokens for _, t in flat),
                                              "rationales": sum(encode.n_tokens(t) > encode.max_tokens for t in rationales)}
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# Evidence (RAG systems): the quoted sentence and whether it is really in the cited passage. Decided 2026-11-23 (option B): the answer
# is "covered" only if its quote is verified, instead of a yes/no flag. The NLI "support" step was removed 2026-11-24 (failed its check).
# ---------------------------------------------------------------------------------------------------------------------------
MIN_QUOTE_WORDS = 5                                      # a shorter "quote" is too easy to find by chance


def normalise_quote(text):
    """The words of a text in lower case: every run of other characters (punctuation, quote marks, dashes, spaces) becomes one space, so
    a quote that drops a comma or writes "blue gray" for "blue-gray" is still the same words."""
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text or "").lower()).strip()


def quote_in_passage(quote, passage):
    """True when the quote is in the passage word for word (after normalise_quote). A quote with "..." or an ellipsis is checked
    piece by piece, the pieces in order; the words of all pieces together must be at least MIN_QUOTE_WORDS."""
    pieces = [normalise_quote(x) for x in re.split(r"\[?\.\.\.\]?|\u2026", quote or "")]
    pieces = [x for x in pieces if x]
    if not pieces or sum(len(x.split()) for x in pieces) < MIN_QUOTE_WORDS:
        return False
    text, at = f" {normalise_quote(passage)} ", 0
    for piece in pieces:
        found = text.find(f" {piece} ", at)                    # whole words only: no match inside a longer word
        if found < 0:
            return False
        at = found + len(piece) + 1
    return True


def cited_passage(p):
    """The text of the passage a prediction cites in its evidence, or None."""
    ev = p.get("evidence") or {}
    for c in p.get("contexts") or []:
        if ev.get("chunk_id") is not None and str(c.get("chunk_id")) == str(ev["chunk_id"]):
            return c.get("text")
    return None


def evidence_metrics(qa, pred):
    """RAG answers that carry an "evidence" field: {"chunk_id": ..., "quote": "..."}, or null for "Evidence: none".
    An answer is covered when it has a valid diagnosis and a quote that is word for word in the cited passage. No model is used.
    Nothing here uses the true class, except negative rejection and the errors counted among the covered answers."""
    ids = [i for i in qa if i in pred and "evidence" in pred[i]]
    if not ids:
        return None
    y = [gold_class(qa[i]) for i in ids]
    groups = [qa[i].get("group_id", i) for i in ids]
    given, found, answers = [], [], []
    for i in ids:
        p = pred[i]
        code = predicted_class(qa[i], p)
        quote = ((p.get("evidence") or {}).get("quote") or "").strip()
        passage = cited_passage(p)
        g = code != INVALID and bool(quote)
        f = g and passage is not None and quote_in_passage(quote, passage)
        given.append(g)
        found.append(f)
        answers.append((code, "yes" if f else "no"))
    n, valid = len(ids), sum(predicted_class(qa[i], pred[i]) != INVALID for i in ids)
    risk = selective_risk(y, answers)
    irrelevant = [k for k, i in enumerate(ids) if not any(y[k] in (c.get("tags") or []) for c in pred[i].get("contexts") or [])]
    out = {"n_questions": n, "valid_answers": valid,
           "evidence_given": sum(given), "evidence_none": sum(pred[i]["evidence"] is None for i in ids),
           "evidence_unparsed": sum(1 for i in ids if pred[i]["evidence"] is not None and not (pred[i]["evidence"].get("quote") or "").strip()),
           "coverage_verified": round(coverage(y, answers), 4), "coverage_verified_95ci": bootstrap_ci(y, answers, groups, coverage),
           "n_covered": sum(found), "selective_risk_verified": None if risk is None else round(risk, 4),
           "negative_rejection": None, "n_without_relevant_passage": len(irrelevant)}
    if irrelevant:       # a rejection is an explicit "Evidence: none" (evidence null); a missing or unreadable line is not one
        out["negative_rejection"] = round(sum(pred[ids[k]]["evidence"] is None for k in irrelevant) / len(irrelevant), 4)
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# similarity-check (researcher's approval, 2026-11-24): does the semantic similarity carry information at all? The first LLM-only
# results gave 0.921 to 0.932 for four models with intervals of +-0.002, which looks flat. A reasoning argues for the disease the model
# answered, so if the measure is valid it should resemble the reference texts of the ANSWERED disease more than those of the other seven.
# The reasoning is compared with the reference texts of all 8 diseases; it is "identified" when the answered disease has the single
# highest similarity (a tie is a miss). Chance is 1 in 8 = 0.125. Pass mark, set on 2026-11-24 before the first run of this check:
# identified in at least 0.50 of the answers overall, and in at least 0.25 of the answers of every answered disease that has 10 or more
# answers. (My proposal, not a published standard.) It uses no label of the question: only the answer and the reasoning.
#   python -m src.main similarity-check [--results results] [--out results/comparisons/tables/similarity_check.json]
# ---------------------------------------------------------------------------------------------------------------------------
SIMILARITY_PASS = "identified in >= 0.50 of the answers overall and >= 0.25 for every answered class with 10 or more answers"


def similarity_check(qa, pred, reference_texts, encode):
    ids = [i for i in qa if (pred.get(i) or {}).get("rationale") and predicted_class(qa[i], pred[i]) != INVALID]
    if not ids:
        return None
    refs = reference_lists(reference_texts)
    flat = [(c, t) for c in CLASSES for t in refs[c]]
    ref_vec = defaultdict(list)
    for (c, _), v in zip(flat, encode([t for _, t in flat])):
        ref_vec[c].append(v)
    vectors = encode([pred[i]["rationale"] for i in ids])
    hits, per_answered = [], defaultdict(list)
    confusion = {c: {d: 0 for d in [*CLASSES, "tie"]} for c in CLASSES}
    own, others, true_right, true_wrong = [], [], [], []
    for i, h in zip(ids, vectors):
        answered, gold = predicted_class(qa[i], pred[i]), gold_class(qa[i])
        sims = {c: max(cosine(h, r) for r in ref_vec[c]) for c in CLASSES}
        best = max(sims.values())
        winners = [c for c, v in sims.items() if v == best]
        guess = winners[0] if len(winners) == 1 else "tie"
        hits.append(1.0 if guess == answered else 0.0)
        per_answered[answered].append(hits[-1])
        confusion[answered][guess] += 1
        own.append(sims[answered])
        others.append(mean([v for c, v in sims.items() if c != answered]))
        (true_right if answered == gold else true_wrong).append(sims[gold])
    per = {c: {"n": len(v), "identified": round(mean(v), 4)} for c, v in per_answered.items()}
    rate = mean(hits)
    floors_ok = all(v["identified"] >= 0.25 for v in per.values() if v["n"] >= 10)
    return {"n_answers": len(ids), "identified": round(rate, 4), "chance": 0.125,
            "per_answered_class": {c: per[c] for c in CLASSES if c in per},
            "confusion_answered_vs_best_match": confusion,
            "mean_similarity_to_answered_class": round(mean(own), 4), "mean_similarity_to_the_other_seven": round(mean(others), 4),
            "mean_similarity_to_true_class_when_right": round(mean(true_right), 4) if true_right else None,
            "mean_similarity_to_true_class_when_wrong": round(mean(true_wrong), 4) if true_wrong else None,
            "pass_mark": SIMILARITY_PASS, "passed": bool(rate >= 0.50 and floors_ok)}


# ---------------------------------------------------------------------------------------------------------------------------
# Image claims in the text-only view (researcher's decision, 2026-11-24). In that view the model gets no picture, so a reasoning that
# says "dermoscopy reveals ..." or "the image shows ..." describes something that does not exist. Found by a word list, so it is a
# lower bound (a paraphrase is missed) and it can flag a harmless sentence: report it as approximate, never as "the hallucination rate".
# ---------------------------------------------------------------------------------------------------------------------------
_SCOPE = r"(?:dermos?cop\w*|dermatos?cop\w*|trichoscop\w*)"                    # dermoscopy, dermoscopic, dermatoscopic, trichoscopy
IMAGE_CLAIM = re.compile(
    rf"\b(?:on|under|upon)\s+{_SCOPE}"                                                # "on dermoscopy"
    rf"|\b{_SCOPE}\s+(?:reveals?|shows?|displays?|demonstrates?|exhibits?|depicts?|images?|examination|view)\b"   # "dermoscopy reveals"
    r"|\b(?:the|this|these|an?|given|provided|attached|dermoscopic|clinical)\s+(?:images?|pictures?|photos?|photographs?)\b"
    r"|\b(?:images?|pictures?|photos?|photographs?)\s+(?:shows?|reveals?|displays?|demonstrates?|depicts?|exhibits?)\b", re.I)
# "dermoscopic datasets" or "dermoscopic features of BCC" are general knowledge, not a claim about this case: they are not matched
NOT_A_CLAIM = re.compile(r"\b(?:no|not|without|absence|absent|lack(?:s|ing)?|unavailable|cannot|can['’]?t|unable|never|neither|nor|"
                         r"if|would|could|unless|needs?|needed|requires?|required|limited)\b|n['’]t\b", re.I)


def image_claim_sentence(text):
    """The first sentence of the text that describes an image or a dermoscopy without denying or hypothesising it, or None."""
    for x in re.split(r"(?<=[.!?;:])\s+|\n+", text or ""):
        if IMAGE_CLAIM.search(x) and not NOT_A_CLAIM.search(x):
            return x.strip()
    return None


def claims_an_image(text):
    """True when a sentence of the text describes an image or a dermoscopy and does not deny or hypothesise it."""
    return image_claim_sentence(text) is not None


def image_claim_metrics(qa, pred):
    """The share of text-only reasonings that claim an image. A record counts when it says "view": "text_only" (the runs writes it) and has a
    reasoning. Returns None when there is no such record."""
    ids = [i for i in qa if i in pred and pred[i].get("view") == "text_only" and (pred[i].get("rationale") or "").strip()]
    if not ids:
        return None
    flags = [1.0 if claims_an_image(pred[i]["rationale"]) else 0.0 for i in ids]
    groups = [qa[i].get("group_id", i) for i in ids]
    return {"n_text_only_with_reasoning": len(ids), "n_claiming_an_image": int(sum(flags)), "rate": round(mean(flags), 4),
            "rate_95ci": bootstrap_ci(flags, flags, groups, lambda a, b: mean(a)),
            "ids_claiming_an_image": [i for i, f in zip(ids, flags) if f],
            "flagged_sentences": {i: image_claim_sentence(pred[i]["rationale"]) for i, f in zip(ids, flags) if f},   # to check by eye
            "method": "word list; approximate lower bound"}


# =============================================================================================================================
# Diagnosis: accuracy, macro-F1, per-class results, invalid answers and refusals, tokens and time
# =============================================================================================================================

# A refusal: the host flagged it (finish_reason "refusal...") or the reply says so in one of these fixed phrases (set on 2026-11-22
# from a real reply: "I'm sorry, I can't help with identifying or diagnosing images of skin lesions."). Only an invalid answer
# (no usable letter) is checked, so a valid answer that apologises is never counted.
REFUSAL = re.compile(r"\bI(?:['’]m| am) sorry\b|\bI(?: can['’]?t| cannot| can not| won['’]t| am unable to|['’]m unable to"
                     r"| am not able to|['’]m not able to) (?:help|assist|provide|identify|diagnose|answer|comply|interpret"
                     r"|analy[sz]e|evaluate)", re.I)


def invalid_causes(ids, classes, pred):
    """Splits the invalid answers into refusals (REFUSAL) and other invalid answers (a reply without a usable letter)."""
    out = {"refusals": [], "other": []}
    for i, c in zip(ids, classes):
        r = pred.get(i)
        if r is None or c != INVALID:
            continue
        refused = str(r.get("finish_reason") or "").startswith("refusal") or REFUSAL.search(str(r.get("raw_output") or ""))
        out["refusals" if refused else "other"].append(i)
    return {"refusals": len(out["refusals"]), "other_invalid": len(out["other"]), "refusal_ids": out["refusals"],
            "other_invalid_ids": out["other"]}


def per_class(y, p):
    """Precision, recall (= sensitivity), specificity and F1 of each class; an invalid answer is wrong for every class."""
    pairs, n, rows = Counter(zip(y, p)), len(y), {}
    for c in CLASSES:
        tp = pairs[(c, c)]
        fn = sum(v for (t, q), v in pairs.items() if t == c and q != c)
        fp = sum(v for (t, q), v in pairs.items() if t != c and q == c)
        tn = n - tp - fn - fp
        prec, rec = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        rows[c] = {"precision": prec, "recall": rec, "specificity": tn / (tn + fp) if tn + fp else 0.0,
                   "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "support": tp + fn}
    return rows


def accuracy(y, p):
    return sum(t == q for t, q in zip(y, p)) / len(y)


def macro_f1(y, p):
    """The mean F1 over the eight classes (every class weighs the same)."""
    return sum(r["f1"] for r in per_class(y, p).values()) / len(CLASSES)


def diagnosis_metrics(qa, pred, pred_b=None):
    unknown = sorted(set(pred) - set(qa))
    if unknown:
        raise ValueError(f"{len(unknown)} predictions are not in the question file, e.g. {unknown[:3]}")
    ids = list(qa)
    y = [gold_class(qa[i]) for i in ids]
    p = [predicted_class(qa[i], pred.get(i)) for i in ids]
    groups = [qa[i].get("group_id", i) for i in ids]
    rows = per_class(y, p)
    columns = [*CLASSES, INVALID]
    out = {"n_questions": len(ids),
           "missing_predictions": sum(i not in pred for i in ids),
           "invalid_answers": sum(i in pred and c == INVALID for i, c in zip(ids, p)),
           "invalid_answers_by_cause": invalid_causes(ids, p, pred),
           "accuracy": round(accuracy(y, p), 4), "accuracy_95ci": bootstrap_ci(y, p, groups, accuracy),
           "macro_f1": round(macro_f1(y, p), 4), "macro_f1_95ci": bootstrap_ci(y, p, groups, macro_f1),
           "per_class": {c: {k: (v if k == "support" else round(v, 3)) for k, v in r.items()} for c, r in rows.items()},
           "confusion_matrix": {"rows_true": CLASSES, "columns_predicted": columns,
                                "matrix": [[sum(t == r and q == c for t, q in zip(y, p)) for c in columns] for r in CLASSES]}}
    out["chosen_letters"] = {k: v for k, v in sorted(Counter(
        (str((pred.get(i) or {}).get("pred") or "").strip().upper() or "none") if c != INVALID else "invalid"
        for i, c in zip(ids, p)).items())}
    if pred_b is not None:
        out["mcnemar_vs_system_b"] = mcnemar_exact(y, p, [predicted_class(qa[i], pred_b.get(i)) for i in ids])
    return out


def usage_summary(qa, pred):
    """Totals and means of tokens and seconds, for the predictions that report them (None when none does)."""
    out = {}
    for key in ("tokens_in", "tokens_out", "seconds"):
        values = [float(pred[i][key]) for i in qa if i in pred and pred[i].get(key) is not None]
        if values:
            out[key] = {"total": round(sum(values), 2), "mean": round(mean(values), 2), "n": len(values)}
    return out or None


# =============================================================================================================================
# Retrieval (RAG systems): Hit@k, Precision@k, Recall@k, nDCG@k, MRR and the MedCPT relevance scores
# =============================================================================================================================

def relevant_counts(passages):
    """How many corpus passages carry each class tag (the weak labels): the denominator of Recall@k."""
    counts = Counter(t for p in passages for t in set(p.get("tags") or []))
    return {c: counts[c] for c in CLASSES}


CUTOFFS = (1, 3, 5, 10)


def retrieval_metrics(qa, pred, ks=None, n_relevant=None):
    """Hit@k, Precision@k, nDCG@k (and Recall@k with the corpus counts) and MRR. Without ks, k is each of 1, 3, 5 and 10 that is not
    larger than the number of passages the system kept for an answer (researcher's decision, 2026-11-25): a system that keeps 3
    passages has no Precision@5 (it would be divided by 5 with 3 passages) and no Hit@5 (it would repeat Hit@3)."""
    rows = [(qa[i]["gold_label"], pred[i]["contexts"]) for i in qa if i in pred and pred[i].get("contexts")]
    if not rows:
        return None
    if ks is None:
        kept = max(len(ctx) for _, ctx in rows)
        ks = tuple(k for k in CUTOFFS if k <= kept)
    res = {}
    ideal = [1 / math.log2(j + 2) for j in range(max(ks))]
    for k in ks:
        hit, prec, nd, rec = [], [], [], []
        for g, ctx in rows:
            rel = [1 if g in c.get("tags", []) else 0 for c in ctx[:k]]
            hit.append(1.0 if any(rel) else 0.0)
            prec.append(sum(rel) / k)
            if n_relevant:
                rec.append(sum(rel) / n_relevant[g] if n_relevant.get(g) else 0.0)
            nd.append(sum(r / math.log2(j + 2) for j, r in enumerate(rel)) / sum(ideal[:k]))
        res[f"hit@{k}"], res[f"precision@{k}"], res[f"ndcg@{k}"] = (round(mean(v), 4) for v in (hit, prec, nd))
        if n_relevant:
            res[f"recall@{k}"] = round(mean(rec), 4)
    mrr = []
    for g, ctx in rows:
        rel = [1 if g in c.get("tags", []) else 0 for c in ctx]
        mrr.append(next((1 / (j + 1) for j, r in enumerate(rel) if r), 0.0))
    res["mrr"] = round(mean(mrr), 4)
    res["n_questions_with_contexts"] = len(rows)
    return res


def medcpt_scorers():
    """Two scorers built on MedCPT (the same models as the RAG retrieval, src/models/embeddings.py): the cosine of the query and
    passage embeddings, and the cross-encoder score (higher = more relevant; not a probability). Each takes (query, passages) and
    returns one score per passage."""
    from src.models.embeddings import MEDCPT_ARTICLE, MEDCPT_QUERY, medcpt_cross_encoder, medcpt_encoder
    query, article, cross_encoder = medcpt_encoder(MEDCPT_QUERY, 64), medcpt_encoder(MEDCPT_ARTICLE, 512), medcpt_cross_encoder()

    def articles(passages):
        return [[c.get("title") or "", c["text"]] for c in passages]

    def cosine(q, passages):
        qv = query([q])[0]
        return [sum(a * b for a, b in zip(qv, v)) for v in article(articles(passages))]      # unit vectors: the dot product

    def cross(q, passages):
        return cross_encoder(q, articles(passages))

    return {"medcpt_cosine": cosine, "medcpt_cross_encoder": cross}


def relevance_metrics(qa, pred, scorers):
    """Mean relevance of the retrieved passages to the query the system searched with (its 'query', else the question):
    first averaged over the passages of a question, then over the questions."""
    ids = [i for i in qa if i in pred and pred[i].get("contexts")]
    if not ids:
        return None
    scores, source = defaultdict(list), Counter()
    for i in ids:
        query, passages = pred[i].get("query") or qa[i]["question"], pred[i]["contexts"]
        source["query" if pred[i].get("query") else "question"] += 1
        for name, fn in scorers.items():
            scores[name].append(mean(fn(query, passages)))
    return {**{k: round(mean(v), 4) for k, v in scores.items()}, "n_questions": len(ids), "query_used": dict(source)}
