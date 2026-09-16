import io
import json
import logging
import urllib.error

from infrastructure import home_assistant
from infrastructure.home_assistant import HomeAssistant


def _fake_urlopen(requests, body):
    def fake_urlopen(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps(body).encode())

    return fake_urlopen


def test_location_reads_the_supervisor_config(monkeypatch):
    requests = []
    body = {"latitude": 52.1, "longitude": 5.2, "time_zone": "Europe/Amsterdam"}
    monkeypatch.setattr(
        home_assistant.urllib.request, "urlopen", _fake_urlopen(requests, body)
    )

    assert HomeAssistant("secret-token").location() == (52.1, 5.2)
    assert requests[0].full_url == "http://supervisor/core/api/config"
    assert requests[0].get_header("Authorization") == "Bearer secret-token"


def test_set_state_writes_the_entity_and_logs_it(monkeypatch, caplog):
    requests = []
    monkeypatch.setattr(
        home_assistant.urllib.request, "urlopen", _fake_urlopen(requests, {})
    )

    with caplog.at_level(logging.INFO):
        HomeAssistant("secret-token").set_state(
            "binary_sensor.home_optimizer_dhw_status", "on", {"friendly_name": "DHW"}
        )

    request = requests[0]
    assert request.get_method() == "POST"
    assert (
        request.full_url
        == "http://supervisor/core/api/states/binary_sensor.home_optimizer_dhw_status"
    )
    assert json.loads(request.data) == {
        "state": "on",
        "attributes": {"friendly_name": "DHW"},
    }
    assert "wrote binary_sensor.home_optimizer_dhw_status = on" in caplog.text


def test_set_state_without_a_token_only_logs(monkeypatch, caplog):
    """A local run has no Supervisor token: it must not reach Home Assistant."""

    requests = []
    monkeypatch.setattr(
        home_assistant.urllib.request, "urlopen", _fake_urlopen(requests, {})
    )

    with caplog.at_level(logging.INFO):
        HomeAssistant(None).set_state(
            "binary_sensor.home_optimizer_dhw_status", "off", {}
        )

    assert requests == []
    assert "not writing binary_sensor.home_optimizer_dhw_status = off" in caplog.text


def test_a_failed_write_is_logged_not_raised(monkeypatch, caplog):
    def unreachable(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(home_assistant.urllib.request, "urlopen", unreachable)

    with caplog.at_level(logging.WARNING):
        HomeAssistant("secret-token").set_state("sensor.x", "1", {})

    assert "writing sensor.x = 1 failed" in caplog.text
