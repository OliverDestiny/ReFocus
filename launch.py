import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from args_manager import args
from webui import gradio_root, get_custom_head
from gradio import mount_gradio_app
from modules import config, html, constants

# -----------------------------
# Paths
# -----------------------------
root = os.path.dirname(os.path.abspath(__file__))
prompt_helper_root = os.path.join(root, "prompt_helper")
static_dir = os.path.join(prompt_helper_root, "static")
aio_root = os.path.join(prompt_helper_root, "sd-webui-prompt-all-in-one")

# Backend needs this path
sys.path.insert(0, prompt_helper_root)

host = args.listen or "0.0.0.0"
port = args.port or int(os.environ.get("GRADIO_SERVER_PORT", "12345"))

app = FastAPI()

# -----------------------------
# 1. STATIC FILES (must be mounted first)
# -----------------------------
app.mount(
    "/prompt-helper/static",
    StaticFiles(directory=static_dir),
    name="prompt-helper-static",
)

@app.get("/prompt-helper")
@app.get("/prompt-helper/")
def prompt_helper_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

# -----------------------------
# 2. EXTENSION JS
# -----------------------------
@app.get("/sd-webui-prompt-all-in-one-js")
def serve_extension_js():
    return FileResponse(os.path.join(aio_root, "javascript", "main.entry.js"))

# -----------------------------
# 3. BACKEND (preserved exactly)
# -----------------------------
from prompt_helper.app import create_prompt_helper_app
prompt_helper_app = create_prompt_helper_app()

app.mount("/prompt-helper", prompt_helper_app)

# -----------------------------
# 4. ReFocus UI
# -----------------------------
mount_gradio_app(
    app,
    gradio_root,
    path="/",
    favicon_path="assets/favicon.png",
    auth=None,
    blocked_paths=[constants.AUTH_FILENAME],
    allowed_paths=[config.path_outputs],
    css=html.css,
    head=get_custom_head(),
)

# -----------------------------
# 5. Run server
# -----------------------------
print(f"ReFocus UI at http://localhost:{port}/")

import webbrowser
if args.in_browser:
    webbrowser.open(f"http://{host}:{port}/")

uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
