import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class HomeAssistant:
    """Home Assistant, reached through the Supervisor API the add-on may call
    (config.yaml: homeassistant_api). Only the add-on has a Supervisor token;
    without one - a local development run - writes are logged but not sent, so
    such a run never changes the real installation's entities."""

    API_URL = "http://supervisor/core/api"

    def __init__(self, token: str | None) -> None:
        self.token = token

    def location(self) -> tuple[float, float]:
        config = self._request("GET", "/config")

        return float(config["latitude"]), float(config["longitude"])

    def set_state(self, entity_id: str, state: str, attributes: dict[str, Any]) -> None:
        if self.token is None:
            logger.info(
                "Home Assistant: no Supervisor token, not writing %s = %s %s",
                entity_id,
                state,
                attributes,
            )
            return

        # A failed write must not fail the run that produced it: the plan is
        # already saved, and the next run writes again minutes later.
        try:
            self._request(
                "POST",
                f"/states/{entity_id}",
                {"state": state, "attributes": attributes},
            )
        except urllib.error.URLError as error:
            logger.warning(
                "Home Assistant: writing %s = %s failed: %s", entity_id, state, error
            )
            return

        logger.info("Home Assistant: wrote %s = %s %s", entity_id, state, attributes)

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        if self.token is None:
            raise RuntimeError(
                "Home Assistant is only reachable from the add-on (no Supervisor "
                "token)."
            )

        request = urllib.request.Request(
            self.API_URL + path,
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
