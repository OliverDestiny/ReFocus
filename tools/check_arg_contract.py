# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

"""Verify the argument contract between the UI (webui.py builds ctrls) and the worker.

The order table is modules/arg_schema.py; run from the repository root. --verify additionally
imports webui to build the UI and checks the total, and everything after '--' is forwarded to the app.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def print_schema():
    import modules.arg_schema as arg_schema

    for group in arg_schema.GROUPS:
        state = '' if group.enabled is None or group.enabled() else '  [disabled]'
        print(f'  {group.name:20s} {len(group.args):3d}  {", ".join(group.args)}{state}')
    print(f'  {"TOTAL":20s} {arg_schema.total():3d}')


def verify_ui():
    """Import webui to build the UI; its per-group assertions fire at import time."""
    import modules.arg_schema as arg_schema

    try:
        import modules.webui as webui
    except Exception as exc:
        print(f'FAIL: could not import webui, the contract was not verified: '
              f'{type(exc).__name__}: {exc}')
        return 1

    ctrls = getattr(webui, 'ctrls', None)
    if ctrls is None:
        print('FAIL: webui has no ctrls (the module layout may have changed).')
        return 1

    expected = arg_schema.total() + 1  # +1 for currentTask
    if len(ctrls) != expected:
        print(f'FAIL: ctrls={len(ctrls)}, expected {expected} '
              f'(schema {arg_schema.total()} + currentTask).')
        return 1

    print(f'OK: ctrls={len(ctrls)} matches the schema '
          f'({arg_schema.total()} parameters + currentTask).')
    return 0


def main():
    argv = sys.argv[1:]
    if '--' in argv:
        split = argv.index('--')
        tool_args, app_args = argv[:split], argv[split + 1:]
    else:
        tool_args, app_args = argv, []

    # args_manager parses sys.argv at import time; keep only the arguments forwarded to the app
    sys.argv = [sys.argv[0]] + app_args

    print('Argument table (modules/arg_schema.py):')
    print_schema()

    if '--verify' in tool_args:
        print()
        return verify_ui()

    print('\n(Add --verify to also import webui and check the total.)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
