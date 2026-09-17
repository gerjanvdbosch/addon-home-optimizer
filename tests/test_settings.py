import pytest

from app import settings


@pytest.fixture
def no_dotenv(monkeypatch):
    monkeypatch.setattr(settings, "load_dotenv", lambda: None)


def test_load_settings_requires_a_location(monkeypatch, no_dotenv):
    """Without a location the solar position is wrong for every reading - that
    must fail at startup, not fall back to some default place."""

    monkeypatch.delenv("LATITUDE", raising=False)
    monkeypatch.delenv("LONGITUDE", raising=False)

    with pytest.raises(ValueError, match="LATITUDE"):
        settings.load_settings()


def test_load_settings_reads_the_location_from_the_environment(monkeypatch, no_dotenv):
    monkeypatch.setenv("LATITUDE", "53.2")
    monkeypatch.setenv("LONGITUDE", "6.5")

    loaded = settings.load_settings()

    assert (loaded.latitude, loaded.longitude) == (53.2, 6.5)
