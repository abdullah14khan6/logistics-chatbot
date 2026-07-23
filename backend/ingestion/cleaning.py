import re


_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTILINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _MULTISPACE_RE.sub(" ", normalized)
    normalized = "\n".join(line.strip() for line in normalized.splitlines())
    normalized = _MULTILINE_RE.sub("\n\n", normalized)
    return normalized.strip()

