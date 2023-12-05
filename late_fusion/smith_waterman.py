from typing import List, Dict, Tuple

import swalign

swalign_vocab = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "$",
    "%",
    "&",
    "/",
    "(",
    ")",
    "=",
    "?",
    "¿",
    "*",
    "<",
    ">",
    "+",
    "#",
    "{",
    "}",
    ";",
    ":",
    "^",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


# Utility function for converting a string sequence to a compatible swalign string
def swalign_preprocess(r: List[str], q: List[str]) -> Tuple[str, str, Dict[str, str]]:
    current_vocab = sorted(set(r + q))
    assert len(current_vocab) < len(swalign_vocab)
    w2swa = dict(zip(current_vocab, swalign_vocab))
    swa2w = dict(zip(swalign_vocab, current_vocab))
    r = ["¡"] + [w2swa[i] for i in r] + ["!"]
    q = ["¡"] + [w2swa[i] for i in q] + ["!"]
    return "".join(r), "".join(q), swa2w


# This is a modified version of the original dump() method for the Alignment class of the swalign library
# We have modified it to obtain (in the following order) the query, the matches, and the reference sequences; all of them have the same length
# Matches is a string that contains either "|" if sequences match on a token, or "." if they disagree,
# or " " if one of them misses a token (in this case the token "-" is included at such position in the corresponding sequence)
def dump(alignment: swalign.Alignment) -> Tuple[str, str, str]:
    i = alignment.r_pos
    j = alignment.q_pos

    q = ""
    m = ""
    r = ""
    qlen = 0
    rlen = 0

    for count, op in alignment.cigar:
        if op == "M":
            qlen += count
            rlen += count
            for k in range(count):
                q += alignment.orig_query[j]
                r += alignment.orig_ref[i]
                if alignment.query[j] == alignment.ref[i] or (
                    alignment.wildcard
                    and (
                        alignment.query[j] in alignment.wildcard
                        or alignment.ref[i] in alignment.wildcard
                    )
                ):
                    m += "|"
                else:
                    m += "."
                i += 1
                j += 1
        elif op == "D":
            rlen += count
            for k in range(count):
                q += "-"
                r += alignment.orig_ref[i]
                m += " "
                i += 1
        elif op == "I":
            qlen += count
            for k in range(count):
                q += alignment.orig_query[j]
                r += "-"
                m += " "
                j += 1
        elif op == "N":
            q += "-//-"
            r += "-//-"
            m += "    "

    while q and r and m:
        return q, m, r


# Utility function for adapting the probility sequence after the swalign computation to be able to obtain the final alignment
def preprocess_prob(s: str, prob: List[float]) -> List[float]:
    new_prob = prob.copy()
    count = 0
    for id, v in enumerate(s):
        if v == "¡" or v == "!":
            new_prob.insert(id + count, 1)
            count += 1
        elif v == "-":
            new_prob.insert(id + count, 0)
            count += 1
    return new_prob


# Utility function for obtaining the final alignment sequence based on the fixed fusion policy
def get_alignment(
    q: str, m: str, r: str, q_prob: List[float], r_prob: List[float]
) -> str:
    alignment = ""
    for qv, mv, rv, qv_prob, rv_prob in zip(q, m, r, q_prob, r_prob):
        # There are three possible scenarios:
        # 1) Both sequences match on a token (mv == "|", qv == rv) -> included
        # 2) Sequences disagree on a token (mv == ".", qv != rv) -> include that of the highest probability
        # 3) A sequence misses a token (mv == " ", (qv or rv) == "-")-> include that of the other
        if mv == "|":
            # Scenario 1
            assert qv == rv
            alignment += qv
        elif mv == ".":
            # Scenario 2
            assert qv != rv
            alignment += qv if qv_prob >= rv_prob else rv
        elif mv == " ":
            # Scenario 3
            assert qv == "-" or rv == "-"
            alignment += qv if rv == "-" else rv
    return alignment


# Utility function for undoing the swalign preprocess
def undo_swalign_preprocess(alignment: str, swa2w: Dict[str, str]) -> List[str]:
    return [swa2w[i] for i in alignment if i not in ["¡", "!"]]
