"""AER Go2 training package.

The top-level package is named `gym` per repository convention. It also exposes
minimal `Env` and `Wrapper` classes used by the legacy Isaac Gym task code so
internal imports do not depend on the external OpenAI Gym package name.
"""

from __future__ import annotations

import os

MINI_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MINI_GYM_ENVS_DIR = os.path.join(MINI_GYM_ROOT_DIR, 'gym', 'envs')


class Env:
    """Minimal environment base compatible with the methods used in this repo."""

    metadata = {}
    reward_range = (-float("inf"), float("inf"))

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        pass


class Wrapper(Env):
    """Minimal wrapper forwarding unknown attributes to the wrapped env."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()
