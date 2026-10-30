"""Deck files: 60 card IDs, one per line (the `deck.csv` the competition expects)."""
from pathlib import Path

DECK_SIZE = 60


class DeckError(ValueError):
    pass


def validate_deck(deck: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Check what can be checked without card data; the engine rejects unknown IDs itself."""
    cards = tuple(deck)
    if len(cards) != DECK_SIZE:
        raise DeckError(f"deck has {len(cards)} cards, expected {DECK_SIZE}")
    if not all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in cards):
        raise DeckError("deck must contain non-negative integer card IDs")
    return cards


def load_deck(path: str | Path) -> tuple[int, ...]:
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise DeckError(f"{path}: {exc.strerror or exc}") from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        cards = [int(line) for line in lines]
    except ValueError as exc:
        raise DeckError(f"{path}: {exc}") from exc
    return validate_deck(cards)


def save_deck(deck: list[int] | tuple[int, ...], path: str | Path) -> None:
    cards = validate_deck(deck)
    Path(path).write_text("".join(f"{c}\n" for c in cards))


def default_deck() -> tuple[int, ...]:
    """The sample deck that ships with the engine."""
    from kaggle_environments.envs.cabt.cabt import deck

    return validate_deck(deck)
