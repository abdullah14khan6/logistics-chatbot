from backend.ingestion.pipeline import IngestionPipeline


def test_staff_directory_pages_are_classified_separately() -> None:
    text = (
        "Our Efficient Team Leaders\n"
        "International Imports\nUmer Khan\numer@example.com\n"
        "International Exports\nZahid Ali\nops@example.com"
    )

    assert (
        IngestionPipeline._content_type(text, "Our Efficient Team Leaders")
        == "staff_directory"
    )


def test_service_pages_receive_service_content_type() -> None:
    text = "Air Freight\nPickup, customs support, and door-to-door delivery."

    assert IngestionPipeline._content_type(text, "Air Freight") == "service"
