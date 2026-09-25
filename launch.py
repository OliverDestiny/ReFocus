import os
import sys
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from modules.args_manager import args
from modules.webui import gradio_root, get_custom_head
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
# Port comes from modules/args_manager (ReFocus default 12345, --port to change it).
port = args.port

app = FastAPI()

# -----------------------------
# 0. THEME BOOTSTRAP (must run before Gradio's own scripts)
# -----------------------------
# The resolved theme has to be in the URL before Gradio boots: its ImageEditor paints the canvas
# background once, at mount, from the theme it has resolved by then, and it stays white without it.
# A script inside Gradio's `head` parameter is too late for that (Gradio injects that head after
# DOMContentLoaded, long after the canvas exists), so this middleware puts a small inline script at
# the top of <head> of the served HTML instead. See docs/README_DEV.md, "Theme".
THEME_BOOTSTRAP = """<script>
(function () {
    var explicit = new URLSearchParams(window.location.search).get('__theme');
    if (explicit === 'dark' || explicit === 'light') {
        window.__refocus_theme_resolved = explicit;
        return;
    }
    var theme = (window.__refocus_theme === 'dark' || window.__refocus_theme === 'light')
        ? window.__refocus_theme
        : ((window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light');
    window.__refocus_theme_resolved = theme;
    try {
        var url = new URL(window.location.href);
        url.searchParams.set('__theme', theme);
        window.history.replaceState(null, '', url);
    } catch (e) { /* Gradio resolves the theme itself; the canvas just may not follow it */ }
})();
</script>"""


@app.middleware("http")
async def inject_theme_bootstrap(request: Request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if request.method != "GET" or "text/html" not in content_type:
        return response

    body = b"".join([chunk async for chunk in response.body_iterator])
    if b"<head>" not in body:
        return Response(content=body, status_code=response.status_code,
                        headers=dict(response.headers), media_type="text/html")

    prefix = ''
    if args.theme:
        prefix = f'<script>window.__refocus_theme = "{args.theme}";</script>'
    body = body.replace(b"<head>", b"<head>" + prefix.encode() + THEME_BOOTSTRAP.encode(), 1)
    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    return Response(content=body, status_code=response.status_code,
                    headers=headers, media_type="text/html")

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
