"""
app/pipeline/hazm_negation.py

Clause-aware negation detection using hazm's POS tagger.

Why "clause-aware" and not "whole sentence"? Consider:
    "درد نداره ولی نفس نمی‌تونم بکشم"
The word "نداره" (negated) is real, but it negates "درد" (pain), not the
actual emergency ("نفس نمی‌تونم بکشم") a few words later. If we checked
the WHOLE sentence for any negated verb, this dangerous sentence would
be wrongly suppressed. So instead, we split the sentence into clauses on
connectors like "ولی"/"اما", find which clause contains the matched
keyword, and only look for negation WITHIN that same clause.

Fails safe: if hazm or its POS model isn't available, every function
here returns False (not negated), so the pipeline falls back to the
existing fixed-window check in 2_triage.py -- nothing breaks.

Setup (already done on your machine):
    pip install hazm
    # Your hazm version is older and doesn't support the newer
    # repo_id/model_filename Hugging Face auto-download style, so we use
    # a manually-downloaded model file instead, expected at:
    #     app/pipeline/hazm_resources/pos_tagger.model
    # (you already downloaded and placed it there.)
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_POS_MODEL_PATH = os.path.join(_HERE, "hazm_resources", "pos_tagger.model")

_tagger = None
_hazm_available = None  # None = not checked yet


def _get_tagger():
    """Lazy-load the hazm POS tagger once, from the local model file.
    Returns None if hazm or the model file isn't available."""
    global _tagger, _hazm_available
    if _tagger is not None:
        return _tagger
    if _hazm_available is False:
        return None
    try:
        from hazm import POSTagger
        if not os.path.exists(_POS_MODEL_PATH):
            _hazm_available = False
            return None
        _tagger = POSTagger(model=_POS_MODEL_PATH)
        _hazm_available = True
        return _tagger
    except Exception:
        # hazm not installed, model file missing/incompatible, etc. --
        # fail safe rather than crash the whole pipeline.
        _hazm_available = False
        return None


def hazm_available() -> bool:
    """Call once to check if hazm is properly installed and usable."""
    return _get_tagger() is not None


# Persian negation forms/prefixes on verbs (broader than the plain-word
# list in 2_triage.py, since here we only check tokens hazm already
# tagged as verbs -- lower false-positive risk).
_NEGATION_VERB_PREFIXES = ("نمی", "نخواه", "ندار", "نکرد", "نشد", "نبود", "نیست")


def _is_negated_verb_token(word: str) -> bool:
    return any(word.startswith(p) for p in _NEGATION_VERB_PREFIXES)


# ---------------------------------------------------------------------------
# Clause splitting
# ---------------------------------------------------------------------------
_CLAUSE_CONNECTORS = ["ولی", "اما"]


def split_into_clauses(text: str):
    """
    Splits text into clauses on "ولی"/"اما", returning a list of
    (start_char_index, end_char_index, clause_text) tuples, indices
    relative to the original `text`.
    """
    pattern = "|".join(re.escape(c) for c in _CLAUSE_CONNECTORS)
    parts = []
    last_end = 0
    for m in re.finditer(pattern, text):
        parts.append((last_end, m.start(), text[last_end:m.start()]))
        last_end = m.end()
    parts.append((last_end, len(text), text[last_end:]))
    return parts


def _clause_containing(text: str, char_index: int):
    """Returns the clause_text that contains the given character index."""
    for start, end, clause_text in split_into_clauses(text):
        if start <= char_index < end or (start <= char_index and end == len(text)):
            return clause_text
    return text  # fallback: whole text if index is out of range somehow


def clause_has_negated_verb(clause_text: str) -> bool:
    """True if hazm finds a negated verb within this single clause."""
    tagger = _get_tagger()
    if tagger is None:
        return False
    try:
        from hazm import word_tokenize
        tokens = word_tokenize(clause_text)
        tagged = tagger.tag(tokens)
    except Exception:
        return False
    return any(pos == "VERB" and _is_negated_verb_token(word) for word, pos in tagged)


def is_negated_at(full_text: str, keyword_char_index: int, keyword_text: str = None) -> bool:
    """
    Main entry point: given the full sentence and the character index
    where a dangerous keyword was found, checks whether the CLAUSE
    containing that keyword has a negated verb -- i.e. whether the
    keyword's own clause is being denied, ignoring negation in other,
    unrelated clauses of the same sentence.

    `keyword_text`, if given, is removed from the clause before checking.
    This matters because some keyword phrases themselves contain a
    grammatically-negated verb that IS the danger itself (e.g. "نفس
    نمی‌تونم بکشم" -- "I CAN'T breathe" -- contains "نمی‌تونم", which
    looks like a negated verb but is the emergency, not a denial of it).
    Without removing the keyword's own text first, hazm would wrongly
    treat that as "this clause is negated" and suppress a real emergency.
    """
    clause = _clause_containing(full_text, keyword_char_index)
    if keyword_text:
        clause = clause.replace(keyword_text, " ")
    return clause_has_negated_verb(clause)


if __name__ == "__main__":
    if not hazm_available():
        print("hazm POS tagger not available -- check installation, internet access, and Python version.")
    else:
        tests = [
            ("سکته نکردم ولی دست راستم بی‌حس شده", "بی‌حس شده"),
            ("درد قفسه سینه ندارم فقط خسته‌م", "درد قفسه سینه"),
            ("درد نداره ولی نفس نمی‌تونم بکشم", "نفس نمی‌تونم بکشم"),
        ]
        for text, keyword in tests:
            idx = text.find(keyword)
            result = is_negated_at(text, idx, keyword_text=keyword)
            print(f"{text!r}\n  clause negated? {result}\n")
