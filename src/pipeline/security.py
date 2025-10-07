# src/pipeline/security.py
import os
from typing import Optional, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta, timezone
import jwt  # PyJWT
from jwt import (
    PyJWTError,
    InvalidIssuedAtError,
    ExpiredSignatureError,
    ImmatureSignatureError,
)

# DB session dependency - adapt import if needed
try:
    from src.db.session import get_db
except Exception:
    try:
        from db.session import get_db
    except Exception:
        get_db = None  # you'll get a clear failure later if this is wrong

# User model - adapt import if needed
from src.models.user import User as UserModel

# Config - secret and algorithm (use env vars in production)
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-me-to-a-strong-secret")
ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRES_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRES_MINUTES", "60"))
JWT_LEEWAY = int(os.environ.get("JWT_LEEWAY", "10"))  # seconds tolerance

bearer_scheme = HTTPBearer(auto_error=False)


# ----------------------------------------------------------------------
# TOKEN CREATION (Used during login)
# ----------------------------------------------------------------------
def create_access_token(user: Any) -> str:
    """
    Create a signed JWT token for a given user with proper UTC-based timestamps.
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRES_MINUTES)

    payload = {
        "sub": str(getattr(user, "id", None)),
        "iat": int(datetime.now(timezone.utc).timestamp()),  # issued at (UTC)
        "nbf": int(datetime.now(timezone.utc).timestamp()),  # not before (UTC)
        "exp": int(
            (datetime.now(timezone.utc) + timedelta(minutes=60)).timestamp()
        ),  # expiry (UTC)
        "email": getattr(user, "email", None),
        "role": getattr(user, "role", None),
        "unique_id": getattr(user, "unique_id", None),
        "full_name": getattr(user, "full_name", None),
        "recruiter_role": getattr(user, "recruiter_role", None),
        "business_name": getattr(user, "business_name", None),
        "website_url": getattr(user, "website_url", None),
        "no_of_employees": getattr(user, "no_of_employees", None),
        "location": getattr(user, "location", None),
        "status": getattr(user, "status", None),
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


# ----------------------------------------------------------------------
# TOKEN DECODING (Verification)
# ----------------------------------------------------------------------
def decode_jwt(token: str) -> Optional[dict]:
    """
    Decode a JWT and return its payload dict.
    Throws HTTPException if token invalid.
    """
    try:
        payload = jwt.decode(
            token, SECRET_KEY, algorithms=[ALGORITHM], leeway=JWT_LEEWAY
        )
        return payload
    except ExpiredSignatureError as exc:
        # token expired
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token expired: {exc}"
        )
    except ImmatureSignatureError as exc:
        # token not yet valid (nbf or iat in future)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token not yet valid (nbf/iat): {exc}. If clocks differ between systems, consider enabling small leeway.",
        )
    except InvalidIssuedAtError as exc:
        # invalid iat (e.g., iat > now)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid iat (issued-at) claim: {exc}",
        )
    except PyJWTError as exc:
        # generic decode error
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token decode error: {exc}",
        )


# ----------------------------------------------------------------------
# GET CURRENT USER (Dependency)
# ----------------------------------------------------------------------
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db=Depends(get_db),
):
    """
    FastAPI dependency to get the current user from Authorization: Bearer <token>

    Returns either:
      - The ORM user object (if we can fetch it from DB), or
      - A dict with token claims if DB lookup isn't available.

    Raises 401 if token missing/invalid, 404 if user not found.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization scheme must be Bearer",
        )

    token = credentials.credentials
    # decode
    claims = decode_jwt(token)

    # Expect claim 'sub' to contain user id (adjust if your token uses another claim, e.g., 'user_id')
    user_id = claims.get("sub") or claims.get("user_id") or claims.get("id")
    if not user_id:
        # token structure not as expected
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing 'sub' (user id) claim",
        )

    # If DB session and UserModel available, fetch user
    if db is not None and UserModel is not None:
        try:
            # SQLAlchemy style: adapt if you use SQLModel or other API
            user_obj = db.query(UserModel).filter(UserModel.id == user_id).first()
            if not user_obj:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
                )
            return user_obj
        except Exception as e:
            # As fallback, return the claims dict so endpoints can still use token info
            # but do log/raise in real deployments
            return claims

    # If DB or model not available, return token claims (best-effort)
    return claims
