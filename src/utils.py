from numpy.random import RandomState


def load_random_state(random_state: RandomState | int | None) -> RandomState:
    if isinstance(random_state, RandomState):
        return random_state
    elif isinstance(random_state, int):
        return RandomState(random_state)
    else:
        return RandomState(None)
