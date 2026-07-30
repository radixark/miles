import hmac

from miles.utils.arguments import _RedactedString


def test_tinker_api_key_keeps_value_but_redacts_log_rendering():
    secret = _RedactedString("tml-secret")

    assert hmac.compare_digest(secret, "tml-secret")
    assert str(secret) == "<redacted>"
    assert repr(secret) == "<redacted>"
    assert f"{secret}" == "<redacted>"
