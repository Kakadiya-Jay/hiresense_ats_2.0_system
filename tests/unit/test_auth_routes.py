import json
import re
from src.models.user import User
from src.utils.security import pwd_context  # passlib context used in project
from sqlalchemy.orm import Session


def test_signup_creates_pending_user(client, db_session):
    payload = {
        "recruiter_name": "Test User",
        "recruiter_role": "HR",
        "business_name": "Test Co",
        "website_url": "http://example.com",
        "no_of_employees": "0-25",
        "email": "recruiter1@example.com",
        "phone": "9123456789",
        "password": "StrongPassword12!",
    }

    resp = client.post("/auth/signup", json=payload)
    assert resp.status_code in (200, 202)
    data = resp.json()
    assert "message" in data

    # Check DB for created user with status pending
    user = db_session.query(User).filter(User.email == payload["email"]).first()
    assert user is not None
    assert user.status == "pending"
    assert user.unique_id is not None
    # password stored as hash (not raw)
    assert user.password_hash != payload["password"]
    assert user.password_hash.startswith("$2b$") or user.password_hash.startswith(
        "$2a$"
    )


def test_login_blocked_until_approved(client, db_session):
    # create user directly (pending)
    from uuid import uuid4

    unique_id = str(uuid4())
    password_plain = "AnotherStrong1!"
    hashed = pwd_context.hash(password_plain)
    user = User(
        unique_id=unique_id,
        email="recruiter2@example.com",
        password_hash=hashed,
        full_name="Blocked User",
        role="recruiter",
        status="pending",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    # attempt login - should be forbidden (403)
    resp = client.post(
        "/auth/login", json={"email": user.email, "password": password_plain}
    )
    assert resp.status_code == 403


def test_full_signup_approve_and_login_flow(client, db_session):
    # signup using route
    signup_payload = {
        "recruiter_name": "Approve User",
        "recruiter_role": "Hiring Manager",
        "business_name": "Approve Co",
        "website_url": "http://example2.com",
        "no_of_employees": "25-50",
        "email": "recruiter3@example.com",
        "phone": "9123456780",
        "password": "ComplexPass123!",
    }
    resp = client.post("/auth/signup", json=signup_payload)
    assert resp.status_code in (200, 202)

    user = db_session.query(User).filter(User.email == signup_payload["email"]).first()
    assert user is not None
    assert user.status == "pending"

    # Now simulate admin approving by directly updating DB (tests isolate DB)
    user.status = "approved"
    db_session.add(user)
    db_session.commit()

    # Now login should work
    login_resp = client.post(
        "/auth/login",
        json={"email": signup_payload["email"], "password": signup_payload["password"]},
    )
    assert login_resp.status_code == 200
    body = login_resp.json()
    assert "access_token" in body
    assert isinstance(body["access_token"], str)
    # token looks like JWT (has two dots)
    assert body["access_token"].count(".") == 2
