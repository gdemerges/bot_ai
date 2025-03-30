from fastapi.testclient import TestClient
from api import app
import pytest

client = TestClient(app)

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
    assert "Box réservé le" in response.text

def test_report_absence():
    payload = {"name": "TestUser", "date": "2025-04-01"}
    response = client.post("/report_absence", json=payload)
    assert response.status_code == 200
    assert "Absence enregistrée" in response.text

@pytest.mark.skip(reason="Skipping /ask_agent test due to dependency on external OpenAI API calls")
def test_ask_agent():
    payload = {"message": "Hello", "user_id": "test", "history": []}
    response = client.post("/ask_agent", json=payload)
    assert response.status_code == 200
