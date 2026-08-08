import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def natural_join(items: list[str]) -> str:
    clean = [item for item in items if item]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def limit_words(text: str, maximum: int) -> str:
    matches = list(re.finditer(r"\S+", text))
    if maximum <= 0 or len(matches) <= maximum:
        return text
    cutoff = matches[maximum - 1].end()
    prefix = text[:cutoff].rstrip()
    sentence_ends = [
        match.end()
        for match in re.finditer(r"[.!?](?:\s|$)", prefix)
        if match.end() >= cutoff * 0.6
    ]
    if sentence_ends:
        return prefix[: sentence_ends[-1]].strip()
    return prefix.rstrip(" ,;:") + "..."


class ResponseSanitizer:
    REPLACEMENTS = {
        "According to the retrieved company context, ": "",
        "According to the retrieved context, ": "",
        "According to the company data, ": "",
        "Based on the PDF, ": "",
        "Retrieved company context": "Available company information",
        "retrieved company context": "available company information",
        "retrieved context": "available information",
        "Pinecone": "internal systems",
        "RAG": "internal systems",
    }

    def sanitize(self, response: str, allowed_emails: set[str]) -> str:
        cleaned = response
        for old, new in self.REPLACEMENTS.items():
            cleaned = cleaned.replace(old, new)
        return self._remove_unauthorized_emails(cleaned, allowed_emails).strip()

    def _remove_unauthorized_emails(
        self,
        response: str,
        allowed_emails: set[str],
    ) -> str:
        lines = []
        for line in response.splitlines():
            emails = EMAIL_RE.findall(line)
            if not emails:
                lines.append(line)
                continue
            if all(email.lower() in allowed_emails for email in emails):
                lines.append(line)
                continue
            redacted = EMAIL_RE.sub(
                lambda match: (
                    match.group(0)
                    if match.group(0).lower() in allowed_emails
                    else ""
                ),
                line,
            ).strip(" ,;:-")
            if redacted and not redacted.lower().endswith(("at", "email")):
                lines.append(redacted)
        return "\n".join(lines)
