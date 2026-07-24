import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path

from backend.config.settings import Settings
from backend.knowledge.company_profile import load_company_profile


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


REQUIRED_PACKAGES = (
    "fastapi",
    "streamlit",
    "langchain_core",
    "langchain_groq",
    "langchain_huggingface",
    "pinecone",
    "fitz",
    "pytesseract",
    "PIL",
    "sentence_transformers",
)


def run_preflight(settings: Settings) -> list[CheckResult]:
    return [
        *_check_packages(),
        _check_tesseract(settings),
        _check_env(settings),
        _check_company_profile(settings),
        _check_data_dir(settings.data_dir),
    ]


def has_failures(results: list[CheckResult]) -> bool:
    return any(not result.ok for result in results)


def _check_packages() -> list[CheckResult]:
    results = []
    for package in REQUIRED_PACKAGES:
        results.append(
            CheckResult(
                name=f"python package: {package}",
                ok=importlib.util.find_spec(package) is not None,
                detail="installed" if importlib.util.find_spec(package) else "missing",
            )
        )
    return results


def _check_tesseract(settings: Settings) -> CheckResult:
    configured = settings.tesseract_cmd
    if configured:
        path = Path(configured)
        return CheckResult(
            name="tesseract",
            ok=path.exists(),
            detail=str(path) if path.exists() else f"configured path not found: {path}",
        )

    detected = shutil.which("tesseract")
    return CheckResult(
        name="tesseract",
        ok=detected is not None,
        detail=detected or "not found on PATH; set TESSERACT_CMD in .env",
    )


def _check_env(settings: Settings) -> CheckResult:
    missing = []
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    if not settings.pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if not settings.pinecone_index_name and not settings.pinecone_host:
        missing.append("PINECONE_INDEX_NAME or PINECONE_HOST")
    if not settings.tracking_url:
        missing.append("TRACKING_URL")
    if missing:
        return CheckResult(name=".env", ok=False, detail=f"missing: {', '.join(missing)}")
    return CheckResult(name=".env", ok=True, detail="required runtime values present")


def _check_data_dir(data_dir: Path) -> CheckResult:
    if not data_dir.exists():
        return CheckResult(name="data PDFs", ok=False, detail=f"missing directory: {data_dir}")
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        return CheckResult(name="data PDFs", ok=False, detail=f"no PDFs found in {data_dir}")
    return CheckResult(name="data PDFs", ok=True, detail=f"{len(pdfs)} PDF(s) found")


def _check_company_profile(settings: Settings) -> CheckResult:
    path = settings.company_profile_path
    if not path.exists():
        return CheckResult(
            name="company profile",
            ok=False,
            detail=f"missing file: {path}",
        )
    try:
        profile = load_company_profile(path)
        profile.resolve_contact(profile.default_contact_role)
    except Exception as exc:
        return CheckResult(
            name="company profile",
            ok=False,
            detail=f"invalid profile: {exc}",
        )
    return CheckResult(
        name="company profile",
        ok=True,
        detail=(
            f"{len(profile.services_offered)} services and "
            f"{len(profile.contacts)} public contacts loaded"
        ),
    )
