from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_read_reservations():
    response = client.get("/reservations")
    assert response.status_code == 200

def test_read_absences():
    response = client.get("/absences")
    assert response.status_code == 200
