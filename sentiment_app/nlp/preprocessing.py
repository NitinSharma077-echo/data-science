import html
import re


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#(\w+)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
ELONGATED_RE = re.compile(r"(.)\1{2,}")
NON_TEXT_RE = re.compile(r"[^a-z0-9_!'?]+")
SPACE_RE = re.compile(r"\s+")

CONTRACTIONS = (
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'s\b"), " is"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'m\b"), " am"),
)

EMOTICONS = (
    (re.compile(r"(:-\)|:\)|=\)|:d|:-d)", re.IGNORECASE), " happyface "),
    (re.compile(r"(:-\(|:\(|=\(|:'\()", re.IGNORECASE), " sadface "),
    (re.compile(r"(;-?\)|;d)", re.IGNORECASE), " winkface "),
)


def normalize_text(text):
    """Normalize tweet/review text without removing sentiment-bearing negations."""
    text = html.unescape(str(text or "")).lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" urltoken ", text)
    text = USER_RE.sub(" usertoken ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)

    for pattern, replacement in EMOTICONS:
        text = pattern.sub(replacement, text)
    for pattern, replacement in CONTRACTIONS:
        text = pattern.sub(replacement, text)

    text = ELONGATED_RE.sub(r"\1\1", text)
    text = NON_TEXT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()
