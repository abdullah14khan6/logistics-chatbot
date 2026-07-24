import uvicorn

from backend.config.settings import get_settings
from backend.utils.logging import configure_logging


def main() -> None:
    configure_logging()
    settings = get_settings()
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.uvicorn_reload,
    )


if __name__ == "__main__":
    main()
