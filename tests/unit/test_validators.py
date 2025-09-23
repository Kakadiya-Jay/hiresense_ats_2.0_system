import pytest
from src.utils import validators
from src.utils.security import validate_password_policy


def test_phone_validator_valid_numbers():
    assert validators.validate_phone("9999999999")
    assert validators.validate_phone("+911234567890")


def test_phone_validator_invalid_numbers():
    assert not validators.validate_phone("abcd1234")
    assert not validators.validate_phone("12345")  # too short


def test_name_mixed_case():
    assert validators.validate_name_mixed_case("John Doe")
    assert validators.validate_name_mixed_case("aB")
    assert not validators.validate_name_mixed_case("john")  # no uppercase
    assert not validators.validate_name_mixed_case("JOHN")  # no lowercase


def test_password_policy_valid_and_invalid():
    # valid: length >=12, has upper, lower, digit, special
    ok, _ = validate_password_policy("StrongPass12!")
    assert ok

    # too short
    ok, msg = validate_password_policy("A1!a")
    assert not ok

    # missing digit
    ok, msg = validate_password_policy("NoDigitPassword!")
    assert not ok

    # contains illegal char '='
    ok, msg = validate_password_policy("ValidPass12=")
    assert not ok
