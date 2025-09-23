from sqlalchemy import Column, String, Integer, Enum, BigInteger, TIMESTAMP, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import ENUM, VARCHAR, CHAR
from src.db.session import Base


class User(Base):
    __tablename__ = "users"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    unique_id = Column(CHAR(36), unique=True, nullable=False)
    email = Column(VARCHAR(255), unique=True, nullable=False)
    password_hash = Column(VARCHAR(255), nullable=False)
    full_name = Column(VARCHAR(200), nullable=False)
    role = Column(ENUM("recruiter", "admin"), nullable=False, default="recruiter")
    recruiter_role = Column(VARCHAR(100), nullable=True)
    business_name = Column(VARCHAR(255), nullable=True)
    website_url = Column(VARCHAR(512), nullable=True)
    linkedin_url = Column(VARCHAR(512), nullable=True)
    no_of_employees = Column(
        ENUM("0-25", "25-50", "50-100", "100-300", "300+"), nullable=True
    )
    phone = Column(VARCHAR(30), nullable=True)
    status = Column(
        ENUM("pending", "approved", "rejected", "disabled"),
        nullable=False,
        default="pending",
    )
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    approved_by = Column(BigInteger, nullable=True)
    approved_at = Column(TIMESTAMP, nullable=True)
    verification_doc_path = Column(VARCHAR(1024), nullable=True)
