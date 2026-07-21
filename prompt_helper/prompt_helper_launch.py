import os
import sys
import subprocess
import atexit
import signal

prompt_helper_process = None

def start_prompt_helper():
    """启动 prompt_helper 独立服务"""
    global prompt_helper_process
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向项目根目录
    prompt_helper_path = os.path.join(base_dir, 'prompt_helper')
    app_py = os.path.join(prompt_helper_path, 'app.py')
    
    if not os.path.exists(app_py):
        print("[Prompt Helper] app.py not found, skipping start.")
        return
    
    print("[Prompt Helper] Starting service...")
    # Windows 下不显示新控制台窗口
    CREATE_NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0
    prompt_helper_process = subprocess.Popen(
        [sys.executable, app_py],
        cwd=prompt_helper_path,
        stdout=None,          # 输出到主控制台
        stderr=None,
        creationflags=CREATE_NO_WINDOW
    )
    print(f"[Prompt Helper] Started with PID {prompt_helper_process.pid}")

def stop_prompt_helper():
    """停止 prompt_helper 服务"""
    global prompt_helper_process
    if prompt_helper_process is None:
        return
    if prompt_helper_process.poll() is None:  # 进程还在运行
        print("[Prompt Helper] Stopping service...")
        prompt_helper_process.terminate()
        try:
            prompt_helper_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("[Prompt Helper] Force killing...")
            prompt_helper_process.kill()
        print("[Prompt Helper] Service stopped.")
    prompt_helper_process = None

# 注册退出清理（无论正常退出还是异常退出均执行）
atexit.register(stop_prompt_helper)

# 捕获 Ctrl+C 和终止信号
def signal_handler(sig, frame):
    stop_prompt_helper()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)