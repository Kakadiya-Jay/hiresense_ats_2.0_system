# src/api/routes/auth.py
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, requests
from fastapi import Request
from pydantic import BaseModel, EmailStr
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
import os, jwt
from datetime import timedelta
import urllib.parse

router = APIRouter(tags=["auth"], prefix="/auth")


# Config (use settings where possible)
JWT_SECRET = (
    getattr(settings, "JWT_SECRET", None)
    or getattr(settings, "SECRET_KEY", None)
    or os.getenv("JWT_SECRET", "hiresense-default-secret")
)
JWT_ALGORITHM = "HS256"
PWD_RESET_EXPIRES_MINUTES = int(getattr(settings, "PWD_RESET_EXPIRES_MINUTES", 15))


# --- Pydantic schemas used locally ---
class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


# --- helper functions ---
def create_pwd_reset_token(
    email: str, expires_minutes: int = PWD_RESET_EXPIRES_MINUTES
) -> str:
    payload = {
        "sub": email,
        "purpose": "pwd_reset",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=expires_minutes),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    # jwt.encode returns str in pyjwt >= 2.x; ensure bytes->str
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    # URL-encode the token for safe embedding into query param
    return urllib.parse.quote_plus(token)


def decode_pwd_reset_token(token_quoted: str) -> dict:
    # token_quoted may be URL-encoded if coming from query param; decode first
    try:
        token = urllib.parse.unquote_plus(token_quoted)
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    # ensure purpose
    if payload.get("purpose") != "pwd_reset":
        raise HTTPException(status_code=400, detail="Invalid reset token")
    return payload


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
            f"Unique ID: {unique_id}\n"
            f"Name: {payload.recruiter_name}\n"
            f"Email: {payload.email}\n"
            f"Company: {payload.business_name}\n"
            f"Recruiter Role: {payload.recruiter_role}\n"
            f"Website: {payload.website_url}\n"
            f"Number of Employees: {payload.no_of_employees}\n"
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

    # Must be approved by admin
    if user.status != "approved":
        raise HTTPException(
            status_code=403, detail="Account not approved yet or is disabled."
        )

    # Security gap fix: ensure is_active flag is True
    if getattr(user, "is_active", True) is False:
        # treat as deactivated
        raise HTTPException(status_code=403, detail="Account has been deactivated.")

    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect email or password.")

    token = create_access_token(user)
    return {"access_token": token, "token_type": "bearer"}


# Adjust these imports to match project layout if necessary:
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


# --- New endpoints: forgot-password and reset-password ---


@router.post("/forgot-password", status_code=200)
def forgot_password(
    request: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Sends password reset email with short-lived JWT token.
    Uses existing send_email util to keep consistent project email behaviour.
    """
    user = db.query(User).filter(User.email == request.email).first()
    # Do not reveal whether account exists — always return same message
    generic_msg = {
        "message": "If an account exists for this email, a reset link was sent."
    }

    if not user:
        return generic_msg

    # create a short-lived token
    token = create_pwd_reset_token(
        user.email, expires_minutes=PWD_RESET_EXPIRES_MINUTES
    )

    # when creating reset link inside forgot-password
    frontend_base = getattr(settings, "FRONTEND_BASE_URL", "http://localhost:8501")
    reset_path = getattr(
        settings, "FRONTEND_RESET_PATH", "/reset-password"
    )  # OR use ?page=Reset%20Password approach
    # Build link that includes the token as URL-encoded param:
    reset_link = f"{frontend_base.rstrip('/')}{reset_path}?token={token}"
    print(f"Email: {user.email} \n Email Token: {token}\n Reset Link: {reset_link}\n")

    subject = "HireSense — Password reset request"
    body = (
        f"Hello {getattr(user, 'full_name', '')},\n\n"
        f"We received a request to reset your password. If you requested this, open the link below:\n\n"
        f"{reset_link}\n\n"
        f"This link will expire in {PWD_RESET_EXPIRES_MINUTES} minutes.\n\n"
        "If you did not request this, ignore this email.\n\nThanks,\nHireSense"
    )

    # send in background to avoid blocking
    background_tasks.add_task(send_email, user.email, subject, body)
    return generic_msg


@router.post("/reset-password", status_code=200)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Accepts token + new_password. Validates token purpose and expiry, updates DB password.
    """
    # Decode and validate token
    claims = decode_pwd_reset_token(payload.token)
    if claims.get("purpose") != "pwd_reset":
        raise HTTPException(status_code=400, detail="Invalid token purpose.")

    email = claims.get("sub")
    if not email:
        raise HTTPException(status_code=400, detail="Invalid reset token payload.")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=400, detail="User not found.")

    if getattr(user, "is_active", True) is False:
        # do not allow reset on deactivated accounts
        raise HTTPException(
            status_code=403, detail="User account has been deactivated."
        )

    # Validate new password policy
    ok, msg = validate_password_policy(payload.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    # Update password
    hashed = get_password_hash(payload.new_password)
    user.password_hash = hashed
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"message": "Password has been reset successfully."}
