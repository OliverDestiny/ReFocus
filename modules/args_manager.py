import argparse
import os
import sys

from tempfile import gettempdir

import comfy.options


# ---------------------------------------------------------------------------
# 1) 先把 ReFocus 独有的参数从 sys.argv 里摘出来
#
# 核心的 comfy/cli_args.py 在**导入时**就解析 sys.argv（cli_args.py:277-280），
# 而它的 parser 不认识 ReFocus 的参数，直接放进去会被 argparse 当作非法参数拒绝。
# 所以这里先用 parse_known_args 摘走我们认识的，再把剩下的原样交还核心，
# 让核心自己的解析与派生逻辑（fast / high_ram / fp16_unet 等）完整跑一遍。
# ---------------------------------------------------------------------------

_refocus_parser = argparse.ArgumentParser(add_help=False)

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
# 2) 让核心解析剩下的参数（含它自己的全部派生逻辑）
# ---------------------------------------------------------------------------

comfy.options.enable_args_parsing(True)

import comfy.cli_args as args_parser  # noqa: E402  (必须在设置 args_parsing 之后导入)

args = args_parser.args


# ---------------------------------------------------------------------------
# 3) 把 ReFocus 的参数并回同一个 Namespace
#
# 注意：--disable-metadata 核心已有同名参数，直接用核心解析出来的值。
# ---------------------------------------------------------------------------

args.preset = _refocus_args.preset
args.language = _refocus_args.language
args.theme = _refocus_args.theme
args.disable_image_log = _refocus_args.disable_image_log
args.disable_preset_selection = _refocus_args.disable_preset_selection
args.disable_analytics = _refocus_args.disable_analytics

# in_browser：核心没有这个概念，保持 ReFocus 自己的默认（开）
args.in_browser = not _refocus_args.disable_in_browser

# 显存策略：核心的 smart memory（能留就留）默认开启，比旧分支「总是卸载」更快，
# 因此这里保留现代默认。--disable-offload-from-vram 仍然接受，语义为「不主动卸载」，
# 与默认一致，所以它实际上只是兼容性开关。
if _refocus_args.disable_offload_from_vram:
    args.disable_smart_memory = False

if args.disable_analytics:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if getattr(args, "temp_path", None) is None:
    args.temp_path = os.path.join(gettempdir(), 'ReFocus')
