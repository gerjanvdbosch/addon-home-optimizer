import io
import json

from infrastructure import home_assistant
from infrastructure.home_assistant import HomeAssistant


def test_location_reads_the_supervisor_config(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        body = {"latitude": 52.1, "longitude": 5.2, "time_zone": "Europe/Amsterdam"}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(home_assistant.urllib.request, "urlopen", fake_urlopen)

    assert HomeAssistant("secret-token").location() == (52.1, 5.2)
    assert requests[0].full_url == "http://supervisor/core/api/config"
    assert requests[0].get_header("Authorization") == "Bearer secret-token"
