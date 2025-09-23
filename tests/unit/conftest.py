import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# import app and DB models
from src.main import app
from src.db.session import Base
from src.db import session as real_session
from src.db.session import get_db as real_get_db

# Create an in-memory sqlite engine for tests
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Create tables in test DB
Base.metadata.create_all(bind=engine)


# Provide a dependency override for get_db
def get_test_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session")
def client():
    # override the dependency
    app.dependency_overrides[real_get_db] = get_test_db
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db_session():
    """Yields a new SQLAlchemy session for a test and cleans up afterwards."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
