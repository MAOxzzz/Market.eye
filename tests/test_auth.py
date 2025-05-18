# tests/test_auth.py
import os
import pytest
from fastapi.testclient import TestClient
import main
from scripts.init_db import init_db

@ pytest.fixture(autouse=True)
def client(tmp_path, monkeypatch):
    # Setup a temporary database for tests
    test_db = tmp_path / "test_market_eye.db"
    # Path to the schema.sql in your project
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'schema.sql')
    # Override DB_PATH in main to use the temp DB
    monkeypatch.setattr(main, "DB_PATH", str(test_db))
    # Initialize the test database schema
    init_db(db_path=str(test_db), schema_path=schema_path)
    return TestClient(main.app)


def test_signup_success(client):
    response = client.post("/signup", json={"username": "alice", "password": "pass"})
    assert response.status_code == 201
    assert response.json()["message"] == "User created"


def test_signup_duplicate(client):
    client.post("/signup", json={"username": "alice", "password": "pass"})
    response = client.post("/signup", json={"username": "alice", "password": "pass"})
    assert response.status_code == 409


def test_login_success(client):
    client.post("/signup", json={"username": "alice", "password": "pass"})
    response = client.post("/login", json={"username": "alice", "password": "pass"})
    assert response.status_code == 200
    assert response.json()["message"] == "Login successful"


def test_login_failure(client):
    response = client.post("/login", json={"username": "nosuch", "password": "pass"})
    assert response.status_code == 401