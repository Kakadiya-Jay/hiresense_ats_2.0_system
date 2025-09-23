import re
from email_validator import validate_email, EmailNotValidError

PHONE_RE = re.compile(r"^\+?\d{7,15}$")


def validate_email_str(email: str) -> bool:
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False


def validate_phone(phone: str) -> bool:
    return bool(PHONE_RE.match(phone))


def validate_name_mixed_case(name: str) -> bool:
    # original requirement: name should contain lower and upper case letters
    has_upper = any(c.isupper() for c in name if c.isalpha())
    has_lower = any(c.islower() for c in name if c.isalpha())
    return has_upper and has_lower
