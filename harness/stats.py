"""Win-rate statistics."""
import math


def wilson(wins: int, games: int, z: float = 1.96) -> tuple[float, float, float]:
    """Observed rate and Wilson score interval (95% by default)."""
    if games == 0:
        return 0.0, 0.0, 1.0
    p = wins / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)
