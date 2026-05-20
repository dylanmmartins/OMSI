# -*- coding: utf-8 -*-
"""
OMSI/_config.py

Set user-specific temp ray path to be stored in internals.yaml. GUI is
used to get path on first use, then saved for all future runs.

Written March 2026, DMM
"""

import os

_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_FILE = os.path.join(_REPO_ROOT, 'internals.yaml')

# hand-rolled yaml parser just for "key: value" lines, keeps the dependency count low
def _load() -> dict:
    if not os.path.exists(_CONFIG_FILE):
        return {}
    config = {}
    with open(_CONFIG_FILE, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                key, _, value = line.partition(':')
                config[key.strip()] = value.strip()
    return config


def _save(config: dict) -> None:
    with open(_CONFIG_FILE, 'w') as fh:
        fh.write('# fMCSI user path configuration -- auto-generated, do not commit\n')
        fh.write('# Delete this file to re-run path setup prompts.\n')
        for key, value in config.items():
            fh.write(f'{key}: {value}\n')


# pops a native file dialog so the user can pick a directory on first run.
# falls back to the system temp dir if tkinter isnt available (e.g. headless servers)
def _pick_directory(prompt: str) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        messagebox.showinfo(
            'fMCSI -- first-time path setup',
            f'{prompt}\n\nClick OK, then choose a folder.',
            parent=root,
        )

        chosen = filedialog.askdirectory(
            title=prompt.splitlines()[0],
            initialdir=os.path.expanduser('~'),
            parent=root,
            mustexist=False,
        )
        root.destroy()

        if chosen:
            return chosen

        print('[fMCSI] No folder selected; using system temp directory.')

    except Exception as exc:
        print(f'[fMCSI] GUI unavailable ({exc}); using system temp directory.')

    import tempfile
    return tempfile.gettempdir()

def get_path(key: str, prompt: str) -> str:

    config = _load()

    if config.get(key, '').strip():
        return config[key].strip()

    path = _pick_directory(prompt)
    os.makedirs(path, exist_ok=True)

    config[key] = path
    _save(config)
    print(f'[fMCSI] {key} = {path}  (saved to internals.yaml)')

    return path
