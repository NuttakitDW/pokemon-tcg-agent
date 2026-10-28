"""Baseline agent and template: develop the board first, then attack (an attack ends the turn).

Self-contained on purpose: `package` copies this one file into the submission. Module-level
code runs under exec() without `__file__`, and only top-level imports resolve.
"""
from kaggle_environments.envs.cabt.cabt import deck as DECK  # noqa: F401  (picked up by the harness)

SELECT_MAIN = 0

OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_RETREAT, OPT_ATTACK, OPT_END = 12, 13, 14

# Lower rank is tried first; options not listed (e.g. retreat) are never chosen voluntarily.
# Attack is last because it ends the turn.
RANK = {OPT_EVOLVE: 0, OPT_ATTACH: 1, OPT_PLAY: 2, OPT_ABILITY: 3, OPT_ATTACK: 4}
MAX_ACTIONS_PER_TURN = 40  # guards against an option the engine keeps re-offering


def _end_turn_index(options):
    for i, option in enumerate(options):
        if option["type"] == OPT_END:
            return i
    return 0


def agent(obs):
    select = obs["select"]
    if select is None:
        return list(DECK)

    options = select["option"]
    if select["type"] != SELECT_MAIN:
        return list(range(select["maxCount"]))

    if obs["current"]["turnActionCount"] > MAX_ACTIONS_PER_TURN:
        return [_end_turn_index(options)]

    ranked = [(RANK[o["type"]], i) for i, o in enumerate(options) if o["type"] in RANK]
    if not ranked:
        return [_end_turn_index(options)]
    return [min(ranked)[1]]
