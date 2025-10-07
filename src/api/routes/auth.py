from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, requests
from fastapi import Request
from pydantic import BaseModel
from typing import Optional, Any, Dict

from sqlalchemy.orm import Session
import uuid
from src.schemas.auth import RecruiterSignup, LoginRequest, UserResponse
from src.db.session import get_db
from src.models.user import User
from src.utils.security import (
    get_password_hash,
    validate_password_policy,
    verify_password,
)
from src.pipeline.security import create_access_token

from src.utils.validators import validate_phone, validate_name_mixed_case
from src.utils.email_utils import send_email
from src.core.config import settings

router = APIRouter(tags=["auth"], prefix="/auth")


@router.post("/signup", status_code=202)
def signup(
    payload: RecruiterSignup,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # Validation
    ok, msg = validate_password_policy(payload.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    if not validate_phone(payload.phone):
        raise HTTPException(
            status_code=400,
            detail="Phone must contain only numbers and optional leading +, length 7-15.",
        )
    if not validate_name_mixed_case(payload.recruiter_name):
        # optional: you can relax this rule; keep as requested
        raise HTTPException(
            status_code=400,
            detail="Name must contain at least one uppercase and one lowercase character.",
        )

    # check unique email
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")

    unique_id = str(uuid.uuid4())
    hashed = get_password_hash(payload.password)

    user = User(
        unique_id=unique_id,
        email=payload.email,
        password_hash=hashed,
        full_name=payload.recruiter_name,
        role="recruiter",
        recruiter_role=payload.recruiter_role,
        business_name=payload.business_name,
        website_url=str(payload.website_url) if payload.website_url else None,
        no_of_employees=payload.no_of_employees,
        phone=payload.phone,
        status="pending",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # notify admin (background)
    def notify_admin():
        subject = "New recruiter signup request"
        body = (
            f"A new recruiter signup request is pending approval.\n\n"
            f"Name: {payload.recruiter_name}\n"
            f"Email: {payload.email}\n"
            f"Company: {payload.business_name}\n"
            f"Unique ID: {unique_id}\n"
            f"You can review at admin dashboard."
        )
        # In MVP we send to team email. In prod you can broadcast to all admins.
        send_email(settings.HIRESENSE_TEAM_EMAIL, subject, body)

    background_tasks.add_task(notify_admin)

    return {
        "message": "Signup request received. Our team will verify your data within 2 working days."
    }


# Admin Email: admin@hiresense.com
# Admin Password: AdminPassword123!
@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password.")
    if user.status != "approved":
        raise HTTPException(
            status_code=403, detail="Account not approved yet or is disabled."
        )
    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect email or password.")

    token = create_access_token(user)
    return {"access_token": token, "token_type": "bearer"}


# Adjust these imports to match your project layout if necessary:
# - get_db: a dependency that yields a DB session (SQLAlchemy / SQLModel)
# - get_current_user: a dependency that validates the JWT and returns a user id or user object
# - User ORM model: the DB model for users

from src.db.session import get_db


try:
    from src.pipeline.security import get_current_user
except Exception as e:
    raise ImportError(
        "Failed to import get_current_user from src.pipeline.security. "
        "Ensure src/pipeline/security.py exists and defines get_current_user. "
        f"Original error: {e}"
    )


from src.models.user import User as UserModel
from datetime import datetime


class UserOut(BaseModel):
    id: Optional[Any]
    unique_id: Optional[str]
    email: Optional[str]
    full_name: Optional[str]
    username: Optional[str]
    business_name: Optional[str]
    website_url: Optional[str]
    phone: Optional[str]
    verified_doc_path: Optional[str]
    role: Optional[str]
    recruiter_role: Optional[str]
    is_active: Optional[bool]
    created_at: Optional[datetime]

    class Config:
        orm_mode = True


@router.get("/me", response_model=UserOut)
def me(current_user=Depends(get_current_user), db=Depends(get_db)):
    """
    Return the profile JSON for the current authenticated user.

    current_user: either an ORM user object returned by get_current_user,
                  or a claims dict if DB/model not available (security fallback).
    """
    # If current_user is an ORM instance, just map fields
    if (
        hasattr(current_user, "__dict__")
        or UserModel
        and isinstance(current_user, UserModel)
    ):
        u = current_user
        return {
            "id": getattr(u, "id", None),
            "unique_id": getattr(u, "unique_id", None),
            "email": getattr(u, "email", None),
            "full_name": getattr(u, "full_name", None) or getattr(u, "name", None),
            "username": getattr(u, "username", None),
            "business_name": getattr(u, "business_name", None),
            "website_url": getattr(u, "website_url", None),
            "phone": getattr(u, "phone", None),
            "verified_doc_path": getattr(u, "verified_doc_path", None),
            "role": getattr(u, "role", None),
            "recruiter_role": getattr(u, "recruiter_role", None),
            "is_active": getattr(u, "is_active", True),
            "created_at": getattr(u, "created_at", None),
        }

    # If current_user is a dict of claims
    if isinstance(current_user, dict):
        return {
            "id": current_user.get("sub")
            or current_user.get("user_id")
            or current_user.get("id"),
            "unique_id": current_user.get("unique_id"),
            "email": current_user.get("email"),
            "full_name": current_user.get("full_name") or current_user.get("name"),
            "username": getattr(u, "username", None),
            "business_name": current_user.get("business_name"),
            "website_url": current_user.get("website_url"),
            "phone": current_user.get("phone"),
            "verified_doc_path": current_user.get("verified_doc_path"),
            "role": current_user.get("role"),
            "recruiter_role": current_user.get("recruiter_role"),
            "is_active": True,
            "created_at": current_user.get("iat"),
        }

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unable to fetch user profile",
    )
