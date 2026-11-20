#!/usr/bin/env python3
"""Standalone keyboard controller for the Uni-SuperMimic sim2sim demo.

A tiny GTK window: focus it with the mouse and press keys; each key press is
forwarded as one byte to /tmp/uni_sim_keys.fifo, which run_sim.py polls.

  u = L2+Up (起步)   x = R1+X (播放)   d = L2+B (阻尼)   p = L2+Down (回PASSIVE)
"""

from __future__ import annotations

import os
import sys

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk  # noqa: E402

FIFO = os.environ.get("UNI_SIM_FIFO", "/tmp/uni_sim_keys.fifo")
LABEL = {
    "u": "u = L2+↑   起步稳定段 → STANDBY",
    "x": "x = R1+X   播放动作 POLICY",
    "d": "d = L2+B   阻尼 DAMP（任意时刻）",
    "p": "p = L2+↓   回 PASSIVE",
    "n": "n = L2+→   下一个动作（STANDBY 时）",
    "b": "b = L2+←   上一个动作（STANDBY 时）",
    "r": "r = 复位   回到刚启动状态（可退出 DAMP）",
}


class Controller(Gtk.Window):
    def __init__(self, fifo_path: str):
        super().__init__(title="Uni-SuperMimic 遥控器")
        self.set_default_size(360, 200)
        self.set_keep_above(True)
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        import time
        while True:
            try:
                self.fifo = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError:
                time.sleep(0.5)  # wait for the simulator to open the read end
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_border_width(12)
        box.pack_start(Gtk.Label(label="聚焦本窗口后按键（终端同步显示状态）"), False, False, 0)
        for ch in ("u", "x", "d", "p", "n", "b", "r"):
            box.pack_start(Gtk.Label(label=LABEL[ch], xalign=0), False, False, 0)
        self.status = Gtk.Label(label="就绪", xalign=0)
        box.pack_start(self.status, False, False, 0)
        self.add(box)
        self.connect("key-press-event", self.on_key)
        self.connect("destroy", Gtk.main_quit)
        self.show_all()

    def on_key(self, _widget, event):
        ch = chr(event.keyval) if 32 <= event.keyval < 127 else ""
        if ch in LABEL:
            try:
                os.write(self.fifo, ch.encode())
            except OSError:
                pass
            self.status.set_text(f"已发送: {LABEL[ch]}")
        return True


if __name__ == "__main__":
    Controller(sys.argv[1] if len(sys.argv) > 1 else FIFO)
    Gtk.main()
