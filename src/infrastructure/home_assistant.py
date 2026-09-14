import json
import urllib.request


class HomeAssistant:
    """Home Assistant's own configuration, read through the Supervisor API that
    the add-on may call (config.yaml: homeassistant_api)."""

    CONFIG_URL = "http://supervisor/core/api/config"

    def __init__(self, token: str) -> None:
        self.token = token

    def location(self) -> tuple[float, float]:
        request = urllib.request.Request(
            self.CONFIG_URL,
            headers={"Authorization": f"Bearer {self.token}"},
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            config = json.load(response)

        return float(config["latitude"]), float(config["longitude"])
