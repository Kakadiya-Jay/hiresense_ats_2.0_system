import re
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from src.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

PASSWORD_POLICY_REGEX = re.compile(
    r"^(?=.{12,36}$)(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[^\w\s]).*$"
)
ILLEGAL_CHARS_REGEX = re.compile(r"[=\|\(\)\[\]]")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def validate_password_policy(password: str) -> (bool, str): # type: ignore
    if ILLEGAL_CHARS_REGEX.search(password):
        return False, "Password contains illegal characters: = | ( ) [ ]"
    if not PASSWORD_POLICY_REGEX.match(password):
        return False, (
            "Password must be 12-36 characters, contain at least one uppercase, "
            "one lowercase, one digit and one special character."
        )
    return True, ""


def create_access_token(subject: str, data: dict = None) -> str:
    to_encode = {"sub": str(subject)}
    if data:
        to_encode.update(data)
    now = datetime.now()
    expire = now + timedelta(seconds=settings.JWT_ACCESS_EXPIRES_SECONDS)
    to_encode.update({"iat": now, "exp": expire})
    encoded = jwt.encode(
        to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    return encoded
