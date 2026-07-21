import os
import sys

from prompt_helper.prompt_helper_launch import start_prompt_helper

# Local modifications
if "--listen" not in sys.argv:
    sys.argv.append("--listen")
if "--always-high-vram" not in sys.argv:
    sys.argv.append("--always-high-vram")

print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "12345"

from args_manager import args

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

from modules import config
os.environ["U2NET_HOME"] = config.path_inpaint

# start the prompt helper service PORT 17860
start_prompt_helper()

# Start the UI
from webui import *
