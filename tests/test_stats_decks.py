import pytest

from harness.decks import DeckError, default_deck, load_deck, save_deck, validate_deck
from harness.stats import wilson


def test_wilson_empty_sample_is_uninformative():
    assert wilson(0, 0) == (0.0, 0.0, 1.0)


def test_wilson_interval_brackets_rate_and_shrinks_with_n():
    p, low, high = wilson(60, 100)
    assert low < p < high
    _, low_big, high_big = wilson(600, 1000)
    assert (high_big - low_big) < (high - low)


@pytest.mark.parametrize("wins,games", [(0, 10), (10, 10)])
def test_wilson_stays_inside_unit_interval(wins, games):
    _, low, high = wilson(wins, games)
    assert 0.0 <= low <= high <= 1.0


def test_deck_roundtrip(tmp_path):
    deck = default_deck()
    save_deck(deck, tmp_path / "deck.csv")
    assert load_deck(tmp_path / "deck.csv") == deck


def test_load_deck_ignores_blank_lines(tmp_path):
    path = tmp_path / "deck.csv"
    path.write_text("\n".join(["3"] * 60) + "\n\n\n")
    assert len(load_deck(path)) == 60


@pytest.mark.parametrize("cards", [[3] * 59, [3] * 61, [3] * 59 + [-1], [3] * 59 + [True]])
def test_validate_deck_rejects_bad_decks(cards):
    with pytest.raises(DeckError):
        validate_deck(cards)


def test_load_deck_rejects_non_integers(tmp_path):
    path = tmp_path / "deck.csv"
    path.write_text("pikachu\n" * 60)
    with pytest.raises(DeckError):
        load_deck(path)
