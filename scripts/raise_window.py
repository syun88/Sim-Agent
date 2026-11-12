#!/usr/bin/env python3
"""Map, raise and activate an X window by name - for a viewer window that ended up minimized.

Measured 2026-11-10: Genesis' viewer window is created and managed by KWin but carries
`_NET_WM_STATE_HIDDEN`, so pyglet's `XlibWindow._map` waits for a MapNotify that never arrives and the cell
never finishes starting. Mapping it from outside is also the test of whether that state is the whole problem:
here it was not - the compositor was not presenting at all (black frames, panel asleep), so no window could map
regardless. Keep this as a tool, and check `spectacle -b -n -o probe.png` (a fully black frame means the
display, not the app) before blaming the viewer.

Usage: DISPLAY=:0 python scripts/raise_window.py [window name]
"""
from __future__ import annotations

import sys

from Xlib import X, display, protocol


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    name = argv[0] if argv else 'run_genesis_gui.py'
    dpy = display.Display()
    root = dpy.screen().root

    def walk(window):
        try:
            title = window.get_wm_name()
        except Exception:
            title = None
        if title == name:
            yield window
        try:
            children = window.query_tree().children
        except Exception:
            children = []
        for child in children:
            yield from walk(child)

    found = list(walk(root))
    print('windows matching', repr(name), ':', [hex(window.id) for window in found])
    hidden = dpy.intern_atom('_NET_WM_STATE_HIDDEN')
    net_state = dpy.intern_atom('_NET_WM_STATE')
    for window in found:
        event = protocol.event.ClientMessage(window=window, client_type=net_state,
                                             data=(32, [0, hidden, 0, 1, 0]))
        root.send_event(event, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        window.map()
        window.configure(stack_mode=X.Above)
        dpy.sync()
        print(hex(window.id), 'map_state =', window.get_attributes().map_state, '(2 = viewable)')
    return 0 if found else 1


if __name__ == '__main__':
    raise SystemExit(main())
