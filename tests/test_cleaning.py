from backend.ingestion.cleaning import clean_text


def test_clean_text_normalizes_spacing() -> None:
    raw = "  Paramount\t\tLogistics  \r\n\r\n\r\n  Freight   services "

    assert clean_text(raw) == "Paramount Logistics\n\nFreight services"

