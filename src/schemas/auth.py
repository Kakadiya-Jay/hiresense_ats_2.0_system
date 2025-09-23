# hiresense/src/schemas/auth.py
from pydantic import BaseModel, EmailStr, AnyUrl, Field, validator
from typing import Optional

class RecruiterSignup(BaseModel):
    recruiter_name: str = Field(..., min_length=2, max_length=200)
    recruiter_role: str
    business_name: Optional[str]
    website_url: Optional[AnyUrl]
    no_of_employees: Optional[str]
    email: EmailStr
    phone: str
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    unique_id: str
    email: EmailStr
    full_name: str
    role: str
    status: str

    class Config:
        # Pydantic v2 renamed orm_mode -> from_attributes
        try:
            from_attributes = True
        except Exception:
            orm_mode = True
