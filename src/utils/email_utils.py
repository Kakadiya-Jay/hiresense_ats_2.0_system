# hiresense/src/utils/email_utils.py
import smtplib
import socket
import ssl
from email.message import EmailMessage
from time import sleep
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def send_email(
    to_email: str, subject: str, body: str, max_retries: int = 2, timeout: int = 10
):
    """
    Robust email sender that works with MailHog (no auth) and real SMTP (STARTTLS/SSL).
    """
    msg = EmailMessage()
    msg["From"] = settings.HIRESENSE_TEAM_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    host = settings.SMTP_HOST
    port = settings.SMTP_PORT
    user = settings.SMTP_USER or None
    password = settings.SMTP_PASSWORD or None

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Trying to send email to %s via %s:%s (attempt %d)",
                to_email,
                host,
                port,
                attempt,
            )

            if port == 465:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(
                    host=host, port=port, context=context, timeout=timeout
                ) as server:
                    server.set_debuglevel(0)
                    if user and password:
                        server.login(user, password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(host=host, port=port, timeout=timeout) as server:
                    server.set_debuglevel(0)
                    server.ehlo()
                    # Only attempt STARTTLS for non-localhost and if server advertises it
                    try:
                        if host not in ("localhost", "127.0.0.1") and server.has_extn(
                            "starttls"
                        ):
                            server.starttls(context=ssl.create_default_context())
                            server.ehlo()
                    except Exception as e:
                        logger.debug("STARTTLS attempt failed: %s", e)

                    if user and password:
                        server.login(user, password)

                    server.send_message(msg)

            logger.info("Email sent to %s", to_email)
            return True
        except (smtplib.SMTPException, socket.error, ssl.SSLError) as exc:
            logger.exception("Failed to send email on attempt %d: %s", attempt, exc)
            last_exc = exc
            if attempt < max_retries:
                sleep(1 + attempt)
            else:
                break
    raise last_exc
