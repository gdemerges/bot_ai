from fastapi.testclient import TestClient
from api import app
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_db_connection(monkeypatch):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    monkeypatch.setattr("api.get_db_connection", lambda: mock_conn)
    monkeypatch.setattr("api.conn", mock_conn)
    monkeypatch.setattr("api.cursor", mock_cursor)

def test_read_reservations():
    response = client.get("/reservations")
    assert response.status_code == 200

def test_read_absences():
    response = client.get("/absences")
    assert response.status_code == 200

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "API bot IA en ligne" in response.json().get("message", "")

def test_book_box():
    payload = {"date": "2025-04-01", "hour": "10:00", "reserved_by": "TestUser"}
    response = client.post("/book_box", json=payload)
    assert response.status_code == 200
    assert "done" in response.text

def test_report_absence():
    payload = {"name": "TestUser", "date": "2025-04-01"}
    response = client.post("/report_absence", json=payload)
    assert response.status_code == 200
    assert "done" in response.text

@pytest.mark.skip(reason="Skipping /ask_agent test due to dependency on external OpenAI API calls")
def test_ask_agent():
    payload = {"message": "Hello", "user_id": "test", "history": []}
    response = client.post("/ask_agent", json=payload)
    assert response.status_code == 200
