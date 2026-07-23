import uvicorn

from backend.utils.logging import configure_logging


def main() -> None:
    configure_logging()
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
