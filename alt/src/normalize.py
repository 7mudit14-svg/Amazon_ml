# -*- coding: utf-8 -*-
"""Country-agnostic normalisation of business names and addresses.

Tables are generic linguistic normalisation (legal forms, street types, number markers)
for EN / IN / FR; everything entity-specific (Indic token aliases, state aliases) is mined
from the provided data by aliases.py and passed in.
"""
import re
import unicodedata

# ----------------------------------------------------------------------------- Indic scripts
_BRAHMIC = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
_INDIC_RE = re.compile(r"[ऀ-෿]")
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
         0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
         0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
         0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
         0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g",
         0x5B: "z", 0x5C: "d", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_VOW = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo", 0x0B: "ri", 0x0C: "li",
        0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au"}
_MATRA = {0x3E: "aa", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri", 0x44: "ri", 0x45: "e",
          0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au"}
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h"}


def has_indic(s):
    return _INDIC_RE.search(s) is not None


def romanize(tok):
    """Rough romanisation of any Brahmic-script token (shared ISCII layout)."""
    out = []
    pending = False
    for ch in tok:
        o = ord(ch)
        base = None
        for b in _BRAHMIC:
            if b <= o < b + 0x80:
                base = b
                break
        if base is None:
            if pending:
                out.append("a")
                pending = False
            out.append(ch)
            continue
        off = o - base
        if off in _CONS:
            if pending:
                out.append("a")
            out.append(_CONS[off])
            pending = True
        elif off in _MATRA:
            out.append(_MATRA[off])
            pending = False
        elif off == 0x4D:  # virama
            pending = False
        elif off in _VOW:
            if pending:
                out.append("a")
                pending = False
            out.append(_VOW[off])
        elif off in _NASAL:
            if pending:
                out.append("a")
                pending = False
            out.append(_NASAL[off])
        elif 0x66 <= off <= 0x6F:
            if pending:
                out.append("a")
                pending = False
            out.append(str(off - 0x66))
    return "".join(out)  # final inherent vowel dropped (schwa deletion)


_SKEL_SUBS = (("chh", "c"), ("ch", "c"), ("sh", "s"), ("ph", "f"), ("kh", "k"), ("gh", "g"), ("th", "t"),
              ("dh", "d"), ("bh", "b"), ("jh", "j"), ("w", "v"), ("q", "k"), ("z", "j"), ("x", "ks"))


def skeleton(tok):
    """Consonant skeleton of a latin token (vowels dropped except a leading one, doubles collapsed)."""
    t = tok
    for a, b in _SKEL_SUBS:
        t = t.replace(a, b)
    if not t:
        return t
    head = t[0]
    body = re.sub(r"[aeiouyh]", "", t[1:])
    s = head + body
    return re.sub(r"(.)\1+", r"\1", s)


# ----------------------------------------------------------------------------- generic tables
LEGAL_STRONG = {
    "pvt": "pvt", "private": "pvt", "ltd": "ltd", "limited": "ltd", "llc": "llc", "llp": "llp",
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "pllc": "pllc",
    "plc": "plc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "snc": "snc",
    "scop": "scop", "selarl": "selarl", "opc": "opc", "company": "co",
}
LEGAL_WEAK = {"co": "co", "sa": "sa", "ei": "ei", "sci": "sci", "pc": "pc", "lp": "lp",
              "public": "public", "sca": "sca", "gie": "gie"}
LEGAL_CODES = sorted(set(LEGAL_STRONG.values()) | set(LEGAL_WEAK.values()))
LEGAL_BIT = {c: 1 << i for i, c in enumerate(LEGAL_CODES)}
HONORIFIC = {"sri", "shri", "shree", "mr", "mrs", "ms", "dr", "messrs", "the", "m"}

ADDR_MAP = {
    "road": "rd", "street": "st", "str": "st", "saint": "st", "avenue": "ave", "av": "ave", "avn": "ave",
    "drive": "dr", "lane": "ln", "court": "ct", "boulevard": "blvd", "bd": "blvd", "bld": "blvd",
    "bvd": "blvd", "highway": "hwy", "parkway": "pkwy", "circle": "cir", "place": "pl", "plaza": "plz",
    "trail": "trl", "terrace": "ter", "square": "sq", "suite": "ste", "sainte": "ste", "apartment": "apt",
    "floor": "fl", "flr": "fl", "north": "n", "south": "s", "east": "e", "west": "w", "mount": "mt",
    "fort": "ft", "county": "co", "township": "twp", "opposite": "opp", "near": "nr", "building": "bldg",
    "district": "dist", "r": "rue", "impasse": "imp", "allee": "allee", "chemin": "chem", "route": "rte",
    "faubourg": "fg", "centre": "center", "nagar": "ngr",
}
NUM_MARKERS = {"no", "nos", "num", "number", "hno", "hn", "dno", "door", "house", "plot", "flat", "shop",
               "shp", "office", "unit", "apt", "ste", "fl", "pmb", "box", "po", "sno", "sy", "khasra", "kh"}
NULL_COMPONENTS = {"null", "n/a", "na", "none", "nan", "-", ""}

_WEB_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|com|in|net|org|fr|biz|info|io|co|us)\b")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-]{7,}\d")
_ID_RE = re.compile(r"\(?\s*\b(?:id|ref)\s*[:#]\s*\w+\s*\)?")
_DBA_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|a\s*/\s*k\s*/\s*a|aka|f\s*/\s*k\s*/\s*a|fka|formerly known as|trading as|t\s*/\s*a)\b")
_DOTTED_RE = re.compile(r"(?<![a-z0-9])(?:[a-z]\.){1,}(?:[a-z](?![a-z0-9]))?")
_SPLIT_RE = re.compile(r"[\s,;:|/\\()\[\]{}<>\"`~!@#$%^*+=?_.\-–—°º]+")
_TOK_RE = re.compile(r"[a-z0-9]+")
_NUM_RE = re.compile(r"\d+")
_LIG = str.maketrans({"œ": "oe", "æ": "ae", "ß": "ss", "ø": "o", "ł": "l", "đ": "d", "ı": "i",
                      "’": "", "'": "", "‘": "", "&": " and "})


def _strip_accents(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _latinize(s, token_alias):
    """Translate Indic tokens through the learned alias table (fallback: romanise)."""
    if not has_indic(s):
        return s, 0, 0
    parts = _SPLIT_RE.split(s)
    out = []
    n_ind = n_fallback = 0
    for p in parts:
        if not p:
            continue
        if has_indic(p):
            n_ind += 1
            a = token_alias.get(p)
            if a is None:
                n_fallback += 1
                a = romanize(p)
            out.append(a)
        else:
            out.append(p)
    return " ".join(out), n_ind, n_fallback


def _dedot(s):
    return _DOTTED_RE.sub(lambda m: m.group(0).replace(".", ""), s)


# ----------------------------------------------------------------------------- names
def norm_name(raw, token_alias):
    """Return dict with core/full/concat strings, legal bitmask and flags."""
    s = unicodedata.normalize("NFKC", raw).lower().translate(_LIG)
    flags = 0
    domain = ""
    if "|" in s:
        main, _, rest = s.partition("|")
        m = _WEB_RE.search(rest)
        if m:
            domain = m.group(1).replace("-", "")
        s = main
    m = _WEB_RE.search(s)
    if m:
        if not domain:
            domain = m.group(1).replace("-", "")
        rest = (s[:m.start()] + " " + s[m.end():]).strip()
        if len(_TOK_RE.findall(rest)) == 0:
            flags |= 1  # whole name is a website
            s = domain
        else:
            s = rest
    if _PHONE_RE.search(s):
        flags |= 2
        s = _PHONE_RE.sub(" ", s)
    if _ID_RE.search(s):
        flags |= 4
        s = _ID_RE.sub(" ", s)
    if _DBA_RE.search(s):
        flags |= 8
        s = _DBA_RE.sub(" ", s)
    s, n_ind, n_fb = _latinize(s, token_alias)
    s = _strip_accents(_dedot(s))
    toks = _TOK_RE.findall(s)
    n = len(toks)
    legal_mask = 0
    is_legal = [False] * n
    for i, t in enumerate(toks):
        if t in LEGAL_STRONG:
            is_legal[i] = True
    for i, t in enumerate(toks):
        if t in LEGAL_WEAK and not is_legal[i]:
            if i == 0 or i == n - 1 or (i > 0 and is_legal[i - 1]) or (i + 1 < n and is_legal[i + 1]):
                is_legal[i] = True
    core = []
    for i, t in enumerate(toks):
        if is_legal[i]:
            code = LEGAL_STRONG.get(t) or LEGAL_WEAK.get(t)
            legal_mask |= LEGAL_BIT[code]
        else:
            core.append(t)
    while core and core[0] in HONORIFIC and len(core) > 1:
        core.pop(0)
    core = [t for t in core if t != "the"] or core
    while len(core) > 1 and core[-1] == "and":
        core.pop()
    if not core:  # name made only of legal words: keep them
        core = toks[:]
    # collapse immediate duplicates ("vidyalaya vidyalaya")
    dedup = [core[0]] if core else []
    for t in core[1:]:
        if t != dedup[-1]:
            dedup.append(t)
    core = dedup
    legal_str = " ".join(c for c in LEGAL_CODES if legal_mask & LEGAL_BIT[c])
    core_s = " ".join(core)
    return {
        "core": core_s,
        "full": (core_s + " " + legal_str).strip(),
        "concat": "".join(core),
        "legal": legal_mask,
        "flags": flags,
        "domain": domain,
        "n_ind": n_ind,
        "n_fb": n_fb,
        "ntok": len(core),
    }


# ----------------------------------------------------------------------------- addresses
def norm_addr(raw, token_alias, comp_alias):
    """Return dict with normalised address string, alpha-only string, number tuple, flags."""
    s = unicodedata.normalize("NFKC", raw).lower().translate(_LIG)
    s = s.replace("n°", " ").replace("nº", " ")
    comps = []
    n_ind = 0
    for c in s.split(","):
        c = c.strip()
        if c in NULL_COMPONENTS:
            continue
        if comp_alias:
            a = comp_alias.get(c)
            if a is not None:
                c = a
        if has_indic(c):
            c, k, _ = _latinize(c, token_alias)
            n_ind += k
        comps.append(c)
    s = _strip_accents(_dedot(", ".join(comps)))
    toks = _SPLIT_RE.split(s)
    out = []
    alpha = []
    nums = []
    prev = ""
    for t in toks:
        if not t:
            continue
        for u in _TOK_RE.findall(t):
            if u.isdigit():
                if prev in ("box", "pmb"):
                    prev = u
                    continue
                v = u.lstrip("0") or "0"
                out.append(v)
                if v not in nums:
                    nums.append(v)
            elif any(ch.isdigit() for ch in u):  # mixed like 41st, a68, 3rd
                m = _NUM_RE.search(u)
                v = m.group(0).lstrip("0") or "0"
                out.append(u)
                if v not in nums:
                    nums.append(v)
            else:
                u = ADDR_MAP.get(u, u)
                out.append(u)
                if u not in NUM_MARKERS:
                    alpha.append(u)
            prev = u
    return {
        "addr": " ".join(out),
        "alpha": " ".join(alpha),
        "nums": tuple(nums[:8]),
        "empty": len(out) == 0,
        "n_ind": n_ind,
    }


if __name__ == "__main__":
    tests_n = ["Sri Rajshi & Có", "L.L.P. Maps Chemicals", "MAPS CHEMICALS-L.L.P.", "ogletreetelecom.com",
               "SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com",
               "Snd Care - 9520147438", "The Ear Nose & Throat Medicine (ID: 34702)",
               "Ciraorbi Labs formerly known as GD Maison SAS", "(E.U.R.L.) Folies Sante (France)",
               "MÉRIGNAC SPORTIVE (S.A.S.)", "Pvt. EFS Print Ventures Ltd.", "Clm Agro [Limited]",
               "राम मार्केटिंग प्राइवेट लिमिटेड", "Sun पावर Provision", "-- Holloway Peak Inc Seafood",
               "Rájshi-+", "Nantes Club SAS & Co", "Gdmaisonsas.Com", "Rajshi & Co"]
    for t in tests_n:
        print(repr(t), "->", norm_name(t, {}))
    tests_a = ["204 Royal Rd, null, Jamestown, North Carolina", "3659. CLIME ROAD, N/A, COLUMBUS, OH",
               "N°33 R. ÉMILE DREUX, BORDEAUX", "Hn 27 S/5 Ajay Centre Opp Pallavitower Navrangpura, Ahmedabad, GJ",
               "02 BUTLER ST, PO BOX 6507, YATES CENTER, KS", "12288-12290 GENEREUX PLACE, ROOGERS, MN",
               "VENUS HOUSING SOCIETY, S. NO 396/6 DAPODI, PUNE, महाराष्ट्र", "", "5 bis Rue Pierre Dignac"]
    for t in tests_a:
        print(repr(t), "->", norm_addr(t, {}, {}))
    print(romanize("राम"), romanize("मार्केटिंग"), romanize("प्राइवेट"), romanize("लिमिटेड"),
          romanize("सूर्य"), romanize("वेंचर्स"), romanize("ஈஸ்டர்ன்"), romanize("கன்சல்டன்சி"))
    print(skeleton("ventures"), skeleton(romanize("वेंचर्स")), skeleton("surya"), skeleton(romanize("सूर्य")))
