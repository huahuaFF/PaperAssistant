from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_endpoint_initializes_local_state(tmp_path: Path) -> None:
    settings = Settings(
        sqlite_path=tmp_path / "app.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        chroma_path=tmp_path / "chroma",
        paper_storage_path=tmp_path / "papers",
        parsed_storage_path=tmp_path / "parsed",
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "Paper Assistant",
        "environment": "development",
    }
    assert (tmp_path / "app.db").exists()
    assert (tmp_path / "chroma").is_dir()
