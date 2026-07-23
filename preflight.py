from backend.config.settings import get_settings
from backend.utils.preflight import has_failures, run_preflight


def main() -> None:
    results = run_preflight(get_settings())
    for result in results:
        marker = "OK" if result.ok else "FAIL"
        print(f"[{marker}] {result.name}: {result.detail}")
    if has_failures(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

