from backend.config.settings import Settings
from backend.ingestion.pipeline import IngestionPipeline


class FakePineconeClient:
    def __init__(self) -> None:
        self.created = False

    def has_index(self, name: str) -> bool:
        return False

    def create_index(self, **kwargs) -> None:
        self.created = True
        self.create_kwargs = kwargs

    def Index(self, name: str):
        return {"name": name}


def test_missing_pinecone_index_is_created(monkeypatch) -> None:
    fake_client = FakePineconeClient()

    def fake_pinecone(api_key: str):
        return fake_client

    class FakeServerlessSpec:
        def __init__(self, cloud: str, region: str) -> None:
            self.cloud = cloud
            self.region = region

    monkeypatch.setattr("backend.ingestion.pipeline.Pinecone", fake_pinecone)
    monkeypatch.setattr("backend.ingestion.pipeline.ServerlessSpec", FakeServerlessSpec)

    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    pipeline.settings = Settings(
        PINECONE_API_KEY="pinecone",
        PINECONE_INDEX_NAME="logistics-company-rag",
        PINECONE_HOST="",
    )

    index = pipeline._pinecone_index()

    assert index == {"name": "logistics-company-rag"}
    assert fake_client.created
    assert fake_client.create_kwargs["dimension"] == 768
    assert fake_client.create_kwargs["metric"] == "cosine"
