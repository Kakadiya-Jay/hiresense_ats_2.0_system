# You can extend dependencies here (e.g., verify JWT, check roles).
from fastapi import Depends, Header, HTTPException
from jose import jwt
from src.core.config import settings
from src.db.session import get_db
from src.models.user import User


def get_bearer_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid auth header")
    return parts[1]


def get_current_user(token: str = Depends(get_bearer_token), db=Depends(get_db)):
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.id == int(payload.get("sub"))).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user
