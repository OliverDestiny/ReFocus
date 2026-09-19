# This file is part of ReFocus.
# Original work Copyright (c) 2023 lllyasviel (Fooocus) & 2024 ehristoforu (DeFooocus).
# Modified and distributed under the terms of the GNU General Public License v3.0.

"""校验 UI 与 worker 之间的参数契约。

UI（webui.py 组装 ctrls）与 worker（async_worker.py 解析 args）靠位置对齐，
顺序表在 modules/arg_schema.py。本工具是独立入口，用于在任何改动后快速确认两端仍然对齐。

用法（在仓库根目录执行）：
    python tools/check_arg_contract.py                       # 打印参数表
    python tools/check_arg_contract.py --verify              # 额外导入 webui 构建 UI 并校验总数
    python tools/check_arg_contract.py --verify -- --disable-metadata   # '--' 之后转发给应用
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
    """导入 webui 构建 UI。webui 的逐组断言在导入期即会触发。"""
    import modules.arg_schema as arg_schema

    try:
        import webui
    except Exception as exc:
        print(f'FAIL: 导入 webui 失败，参数契约未通过校验：\n  {type(exc).__name__}: {exc}')
        return 1

    ctrls = getattr(webui, 'ctrls', None)
    if ctrls is None:
        print('FAIL: 未能从 webui 取到 ctrls（模块结构可能已变）。')
        return 1

    expected = arg_schema.total() + 1  # +1 是 currentTask
    if len(ctrls) != expected:
        print(f'FAIL: ctrls={len(ctrls)}，期望 {expected}（schema {arg_schema.total()} + currentTask）。')
        return 1

    print(f'OK: ctrls={len(ctrls)} 与 schema 一致（{arg_schema.total()} 个参数 + currentTask）。')
    return 0


def main():
    argv = sys.argv[1:]
    if '--' in argv:
        split = argv.index('--')
        tool_args, app_args = argv[:split], argv[split + 1:]
    else:
        tool_args, app_args = argv, []

    # args_manager 在导入时解析 sys.argv，这里只保留要转发给应用的参数
    sys.argv = [sys.argv[0]] + app_args

    print('参数表（modules/arg_schema.py）：')
    print_schema()

    if '--verify' in tool_args:
        print()
        return verify_ui()

    print('\n（加 --verify 可额外导入 webui 校验总数）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
