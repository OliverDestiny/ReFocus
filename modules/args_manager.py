import argparse
import os
import sys

from tempfile import gettempdir

import comfy.options


# ---------------------------------------------------------------------------
# 1) Strip ReFocus-only flags from sys.argv before the core parses it
# comfy/cli_args.py parses sys.argv at import time (cli_args.py:277-280) and rejects unknown
# flags, so ours are taken out first with parse_known_args and the rest is handed back as is.
# ---------------------------------------------------------------------------

_refocus_parser = argparse.ArgumentParser(add_help=False)

_refocus_parser.add_argument("--port", type=int, default=12345,
                             help="Port of the web UI (default 12345). The core's --port defaults to 8188.")

_refocus_parser.add_argument("--disable-metadata", action='store_true',
                             help="Do not write prompt metadata into generated images.")

_refocus_parser.add_argument("--preset", type=str, default=None,
                             help="Apply specified UI preset.")

_refocus_parser.add_argument("--language", type=str, default='default',
                             help="Translate UI using json files in [language] folder. "
                                  "For example, [--language example] will use [language/example.json] for translation.")

_refocus_parser.add_argument("--disable-offload-from-vram", action="store_true",
                             help="Force keeping models in VRAM when the unload can be avoided.")

_refocus_parser.add_argument("--theme", type=str, default=None,
                             help="launches the UI with light or dark theme")

_refocus_parser.add_argument("--disable-image-log", action='store_true',
                             help="Prevent writing images and logs to hard drive.")

_refocus_parser.add_argument("--disable-analytics", action='store_true',
                             help="Disables analytics for Gradio.")

_refocus_parser.add_argument("--disable-preset-selection", action='store_true',
                             help="Disables preset selection in Gradio.")

_refocus_parser.add_argument("--disable-in-browser", action='store_true',
                             help="Do not open the browser on startup.")

_refocus_args, _remaining_argv = _refocus_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining_argv


# ---------------------------------------------------------------------------
# 2) Let the core parse the remaining arguments (including its own derivation logic)
# ---------------------------------------------------------------------------

comfy.options.enable_args_parsing(True)

import comfy.cli_args as args_parser  # noqa: E402  (must be imported after enabling args_parsing)

args = args_parser.args


# ---------------------------------------------------------------------------
# 3) Merge the ReFocus flags back into the core's Namespace
# Names assigned here override the core's values, defaults included, so every name listed
# below is decided by ReFocus; core-only flags (--listen, --temp-path, ...) stay the core's.
# ---------------------------------------------------------------------------

args.preset = _refocus_args.preset
args.language = _refocus_args.language
args.theme = _refocus_args.theme
args.disable_image_log = _refocus_args.disable_image_log
args.disable_preset_selection = _refocus_args.disable_preset_selection
args.disable_analytics = _refocus_args.disable_analytics
args.port = _refocus_args.port
args.disable_metadata = _refocus_args.disable_metadata

# in_browser: unknown to the core, so keep ReFocus's own default (open the browser)
args.in_browser = not _refocus_args.disable_in_browser

# VRAM policy: the core's smart memory (keep what fits) stays on; it is faster than the old
# branch's always-offload, so --disable-offload-from-vram is accepted but only a compat switch.
if _refocus_args.disable_offload_from_vram:
    args.disable_smart_memory = False

if args.disable_analytics:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if getattr(args, "temp_path", None) is None:
    args.temp_path = os.path.join(gettempdir(), 'ReFocus')
