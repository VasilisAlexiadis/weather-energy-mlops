from dataclasses import dataclass

@dataclass
class LocationConfig:
    latitude: float
    longitude: float
    timezone: str = "Europe/Athens"

DEFAULT_LOCATION = LocationConfig(
    latitude=37.98,  # Athens
    longitude=23.72,
)

DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"
