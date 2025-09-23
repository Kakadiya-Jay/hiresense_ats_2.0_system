from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
import uuid
from src.schemas.auth import RecruiterSignup, LoginRequest, UserResponse
from src.db.session import get_db
from src.models.user import User
from src.utils.security import (
    get_password_hash,
    validate_password_policy,
    verify_password,
    create_access_token,
)
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

    token = create_access_token(
        subject=user.id,
        data={"role": user.role, "unique_id": user.unique_id, "email": user.email},
    )
    return {"access_token": token, "token_type": "bearer"}
