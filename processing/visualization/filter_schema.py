"""Single source of truth for filter control keys and choices."""

from dataclasses import dataclass

FILTER_PREFIX = "filter."
PDH_KEY = "filter.pdh.max"
RCS_KEY = "filter.rcs.min"


@dataclass(frozen=True)
class FilterOption:
    field: str
    value: int
    label: str
    default: bool = False
    enabled: bool = True
    color: str | None = None

    @property
    def key(self) -> str:
        return f"{FILTER_PREFIX}{self.field}.{self.value}"


DYNAMIC_PROPERTY_OPTIONS = (
    FilterOption("dynamic_property", 0, "Moving", True, color="#FF0000"),
    FilterOption("dynamic_property", 1, "Stationary", True, color="#FF7B00"),
    FilterOption("dynamic_property", 2, "Oncoming", True, color="#FFE600"),
    FilterOption("dynamic_property", 3, "Stationary Candidate", color="#00FF00"),
    FilterOption("dynamic_property", 4, "Unknown", True, color="#0000FF"),
    FilterOption("dynamic_property", 5, "Crossing Stationary", True, color="#00FFFF"),
    FilterOption("dynamic_property", 6, "Crossing Moving", True, color="#8400FF"),
    FilterOption("dynamic_property", 7, "Stopped", True, color="#000000"),
)
AMBIGUITY_STATE_OPTIONS = (
    FilterOption("ambiguity_state", 1, "Ambiguous"),
    FilterOption("ambiguity_state", 2, "Staggered Ramp"),
    FilterOption("ambiguity_state", 3, "Unambiguous", True),
    FilterOption("ambiguity_state", 4, "Stationary Candidates", True),
)
_DEFAULT_INVALID = {0x00, 0x04, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0F, 0x10, 0x11}
_DISABLED_INVALID = {0x05, 0x0D}
INVALID_STATE_OPTIONS = tuple(
    FilterOption("invalid_state", value, f"0x{value:X}",
                 value in _DEFAULT_INVALID, value not in _DISABLED_INVALID)
    for value in range(0x12)
)
DYNAMIC_COLORS_BGR = tuple(
    tuple(int(option.color[index:index + 2], 16) for index in (5, 3, 1))
    for option in DYNAMIC_PROPERTY_OPTIONS
)


def parse_filter_key(key: object) -> tuple[str, int | str] | None:
    if not isinstance(key, str) or not key.startswith(FILTER_PREFIX):
        return None
    parts = key.split(".")
    if len(parts) != 3:
        return None
    field, raw = parts[1:]
    if (field, raw) in (("pdh", "max"), ("rcs", "min")):
        return field, raw
    try:
        return field, int(raw, 0)
    except ValueError:
        return None
