from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from src.db.session import get_db
from src.models.user import User
from typing import List
from src.schemas.auth import UserResponse
from src.utils.email_utils import send_email
from src.core.config import settings
from jose import jwt
from src.core.config import settings

router = APIRouter(tags=["admin"], prefix="/admin")

# For MVP, a minimal admin dependency that checks JWT in Authorization header.
from fastapi import Security, Header


def get_bearer_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid auth header")
    return parts[1]


def get_current_admin(
    token: str = Depends(get_bearer_token), db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_id = int(payload.get("sub"))
    user = db.query(User).filter(User.id == user_id).first()
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin permission required")
    return user


@router.get("/pending-signups", response_model=List[UserResponse])
def pending_signups(
    db: Session = Depends(get_db), admin: User = Depends(get_current_admin)
):
    users = db.query(User).filter(User.status == "pending").all()
    return users


@router.post("/approve/{user_id}")
def approve_user(
    user_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.status = "approved"
    user.approved_by = admin.id
    from datetime import datetime

    user.approved_at = datetime.utcnow()
    db.add(user)
    db.commit()
    db.refresh(user)

    def send_success_email():
        subject = "Identify your request as valid recruiter."
        body = (
            f"Hello {user.full_name},\n\n"
            f"Your request has been verified by our team and you are now a valid user for HireSense.\n\n"
            f"Your unique id for our system is: {user.unique_id}\n\n"
            f"You can log in at: {settings.FRONTEND_URL}\n\n"
            f"If you encounter any issues, please contact us via: {settings.CONTACT_US_FORM_URL}\n\n"
            "Best,\nHireSense Team"
        )
        send_email(user.email, subject, body)

    background_tasks.add_task(send_success_email)
    return {"message": "User approved and notified."}


@router.post("/reject/{user_id}")
def reject_user(
    user_id: int,
    reason: str = "",
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.status = "rejected"
    db.add(user)
    db.commit()

    def send_reject_email():
        subject = "Recruiter signup request - rejected"
        body = (
            f"Hello {user.full_name},\n\n"
            f"Your signup request has been reviewed and rejected. Reason: {reason}\n\n"
            f"If you believe this is a mistake, contact us: {settings.CONTACT_US_FORM_URL}\n\n"
            "Best,\nHireSense Team"
        )
        send_email(user.email, subject, body)

    if background_tasks:
        background_tasks.add_task(send_reject_email)
    else:
        send_reject_email()
    return {"message": "User rejected and notified."}
