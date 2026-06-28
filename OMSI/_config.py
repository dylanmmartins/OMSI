# -*- coding: utf-8 -*-
"""
OMSI/_config.py

Read and write user-specific path configuration for Ray and other scratch directories.

Functions
---------
_load
    Read internals.yaml into a dict.
_save
    Write a config dict back to internals.yaml.
_pick_directory
    Show a native file dialog and return the chosen path.
get_path
    Get or set a user path in config; shows dialog on first run.


DMM, March 2026
"""

import os

_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_FILE = os.path.join(_REPO_ROOT, 'internals.yaml')


def _load() -> dict:
    """ Read internals.yaml into a dict.

    Hand-rolled parser for "key: value" lines -- avoids adding PyYAML as a
    dependency. Skips blank lines and comments.

    Returns
    -------
    dict
        Key-value pairs from internals.yaml, or empty dict if file absent.
    """

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
    """ Write config dict back to internals.yaml. """

    with open(_CONFIG_FILE, 'w') as fh:
        fh.write('# fMCSI user path configuration -- auto-generated, do not commit\n')
        fh.write('# Delete this file to re-run path setup prompts.\n')
        for key, value in config.items():
            fh.write('{}: {}\n'.format(key, value))


def _pick_directory(prompt: str) -> str:
    """ Show a native file dialog and return the chosen directory path.

    Falls back to system temp directory if tkinter is not available
    (e.g. headless servers).

    Parameters
    ----------
    prompt : str
        Message shown in the dialog box.

    Returns
    -------
    str
        Path to the chosen directory.
    """

    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        messagebox.showinfo(
            'fMCSI -- first-time path setup',
            '{}\n\nClick OK, then choose a folder.'.format(prompt),
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

        print('[fMCSI] No folder selected -- using system temp directory.')

    except Exception as exc:
        print('[fMCSI] GUI unavailable ({}) -- using system temp directory.'.format(exc))

    import tempfile
    return tempfile.gettempdir()


def get_path(key: str, prompt: str) -> str:
    """ Get or set a user path in config; shows dialog on first run.

    Parameters
    ----------
    key : str
        Config key to look up or store (e.g. 'ray_dir').
    prompt : str
        Message shown if path has not been configured yet.

    Returns
    -------
    str
        Absolute path for the requested key.
    """

    config = _load()

    if config.get(key, '').strip():
        return config[key].strip()

    path = _pick_directory(prompt)
    os.makedirs(path, exist_ok=True)

    config[key] = path
    _save(config)
    print('[fMCSI] {} = {}  (saved to internals.yaml)'.format(key, path))

    return path
