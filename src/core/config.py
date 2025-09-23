# hiresense/src/core/config.py
import os
from dotenv import load_dotenv

# load .env from project root
load_dotenv()


class Settings:
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL")

    # JWT
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    try:
        JWT_ACCESS_EXPIRES_SECONDS: int = int(
            os.getenv("JWT_ACCESS_EXPIRES_SECONDS", "3600")
        )
    except ValueError:
        JWT_ACCESS_EXPIRES_SECONDS: int = 3600

    # SMTP
    SMTP_HOST: str = os.getenv("SMTP_HOST", "localhost")
    try:
        SMTP_PORT: int = int(os.getenv("SMTP_PORT", 1025))
    except ValueError:
        SMTP_PORT: int = 1025
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    HIRESENSE_TEAM_EMAIL: str = os.getenv(
        "HIRESENSE_TEAM_EMAIL", "hiresense@example.com"
    )

    # Frontend
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:8501")
    CONTACT_US_FORM_URL: str = os.getenv(
        "CONTACT_US_FORM_URL", f"{FRONTEND_URL}/contact"
    )


# single settings instance used across app
settings = Settings()
