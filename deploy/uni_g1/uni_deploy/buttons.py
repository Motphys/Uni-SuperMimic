"""Wireless controller key bits and debouncing (Unitree G1 handheld)."""

from __future__ import annotations


class KeyMap:
    R1 = 0
    L1 = 1
    START = 2
    SELECT = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    UP = 12
    RIGHT = 13
    DOWN = 14
    LEFT = 15


def pressed(keys: int, bit: int) -> bool:
    return bool((keys >> bit) & 1)


class DebouncedButtons:
    """3-sample debounce; emits edge events understood by FSM.request."""

    def __init__(self, debounce_samples: int = 3):
        self.n = debounce_samples
        self._stable = {"damp": 0, "up": 0, "x": 0, "down": 0}
        self._count = {"damp": 0, "up": 0, "x": 0, "down": 0}

    def poll(self, keys: int):
        raw = {
            "damp": pressed(keys, KeyMap.L2) and pressed(keys, KeyMap.B),
            "up": pressed(keys, KeyMap.L2) and pressed(keys, KeyMap.UP),
            "x": pressed(keys, KeyMap.R1) and pressed(keys, KeyMap.X),
            "down": pressed(keys, KeyMap.L2) and pressed(keys, KeyMap.DOWN),
            "next": pressed(keys, KeyMap.L2) and pressed(keys, KeyMap.RIGHT),
            "prev": pressed(keys, KeyMap.L2) and pressed(keys, KeyMap.LEFT),
            "reset": pressed(keys, KeyMap.R1) and pressed(keys, KeyMap.A),
        }
        events = []
        for name, value in raw.items():
            self._count.setdefault(name, 0)
            self._stable.setdefault(name, 0)
            self._count[name] = self._count[name] + 1 if value else 0
            if value and self._count[name] >= self.n and self._stable[name] == 0:
                self._stable[name] = 1
                events.append(name)
            elif not value:
                self._stable[name] = 0
        return events


EVENT_TO_FSM = {
    "damp": "damp_anytime",
    "up": "to_start",
    "x": "play",
    "down": "to_passive",
    "reset": "reset",
}

EVENT_CYCLE = {
    "next": +1,
    "prev": -1,
}
