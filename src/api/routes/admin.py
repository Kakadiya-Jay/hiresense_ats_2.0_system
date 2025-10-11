from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Body
from sqlalchemy.orm import Session
from src.db.session import get_db
from src.models.user import User
from typing import List
from src.schemas.auth import UserResponse
from src.utils.email_utils import send_email
from src.core.config import settings
from jose import jwt

from pydantic import BaseModel
from typing import List, Optional, Any, Dict

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

router = APIRouter(tags=["admin"], prefix="/admin")

# For MVP, a minimal admin dependency that checks JWT in Authorization header.
from fastapi import Security, Header

from datetime import datetime


class UserListOut(BaseModel):
    id: Optional[Any]
    unique_id: Optional[str]
    email: Optional[str]
    full_name: Optional[str]
    username: Optional[str]
    role: Optional[str]
    recruiter_role: Optional[str]
    business_name: Optional[str]
    website_url: Optional[str]
    phone: Optional[str]
    verified_doc_path: Optional[str]
    is_active: Optional[bool]
    created_at: Optional[datetime]

    class Config:
        orm_mode = True


class UpdateStatusIn(BaseModel):
    active: bool


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


def _is_admin_user(current_user) -> bool:
    """
    Return True if the current_user has admin privileges.
    Accepts ORM object or claims dict.
    """
    # If get_current_user returns an ORM user use its role/is_admin property
    if hasattr(current_user, "role"):
        return getattr(current_user, "role", "").lower() in (
            "admin",
            "superuser",
            "administrator",
        ) or getattr(current_user, "is_admin", False)
    # If it's a dict
    if isinstance(current_user, dict):
        return (current_user.get("role") or "").lower() in (
            "admin",
            "superuser",
        ) or current_user.get("is_admin") is True
    return False


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
    reason: str = "Your signup request did not meet our criteria. Please contact support for more details.",
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

    background_tasks.add_task(send_reject_email)
    return {"message": "User rejected and notified."}


@router.get("/users", response_model=List[UserListOut])
def list_users(current_user=Depends(get_current_user), db=Depends(get_db)):
    """
    Return list of users for admin.
    """
    # authorization check
    if not _is_admin_user(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )

    # If UserModel is not importable, return a helpful error
    if UserModel is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="UserModel not configured on server",
        )

    # Fetch users from DB
    try:
        # SQLAlchemy ORM query; adjust if you use SQLModel or different API
        users = db.query(UserModel).all()
        result = []
        for u in users:
            result.append(
                {
                    "id": getattr(u, "id", None),
                    "unique_id": getattr(u, "unique_id", None),
                    "email": getattr(u, "email", None),
                    "full_name": getattr(u, "full_name", None)
                    or getattr(u, "name", None),
                    "username": getattr(u, "username", None),
                    "business_name": getattr(u, "business_name", None),
                    "website_url": getattr(u, "website_url", None),
                    "phone": getattr(u, "phone", None),
                    "verified_doc_path": getattr(u, "verified_doc_path", None),
                    "role": getattr(u, "role", None),
                    "recruiter_role": getattr(u, "recruiter_role", None),
                    "is_active": bool(getattr(u, "is_active", True)),
                    "created_at": getattr(u, "created_at", None),
                }
            )
        return result
    except Exception as e:
        # If query failed because of imports, try to fallback if current_user contains info
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch users - adapt admin.py imports \n DB query error: {e}",
        )


@router.put("/users/{user_id}/status", status_code=200)
def update_user_status(
    user_id: str,
    payload: UpdateStatusIn = Body(...),
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Update user's active status (activate/deactivate).
    Body: {"active": true/false}
    """
    if not _is_admin_user(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )

    if UserModel is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="UserModel not configured on server",
        )

    # Validate and update user in DB
    try:
        user_obj = db.query(UserModel).filter(UserModel.id == user_id).first()
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="User model or DB query not available - adapt admin.py imports",
        )

    if not user_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    try:
        # update field name depending on your model (is_active / active)
        if hasattr(user_obj, "is_active"):
            setattr(user_obj, "is_active", payload.active)
        elif hasattr(user_obj, "active"):
            setattr(user_obj, "active", payload.active)
        else:
            # try generic
            setattr(user_obj, "is_active", payload.active)

        db.add(user_obj)
        db.commit()
        db.refresh(user_obj)
        return {
            "id": getattr(user_obj, "id", None),
            "active": payload.active,
            "message": "User status updated",
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to update user status: {e}"
        )
