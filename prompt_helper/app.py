import sys
import os
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
AIO_PATH = os.path.join(BASE_DIR, 'sd-webui-prompt-all-in-one')
sys.path.append(AIO_PATH)

from dotenv import load_dotenv
import uvicorn
import gradio as gr
from gradio import Blocks
from fastapi import FastAPI, Response, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.requests import Request
from typing import Optional, Dict, Any
from scripts.on_app_started import on_app_started
from ph_modules.script_callbacks import app_started_callback
import secrets

def create_prompt_helper_app() -> FastAPI:
    load_dotenv()
    app_port = int(os.environ.get("APP_PORT", 17860))

    app = FastAPI()

    app_username = os.environ.get('APP_USERNAME')
    app_password = os.environ.get('APP_PASSWORD')
    if app_username and app_password and app_username != '' and app_password != '':
        security = HTTPBasic()

        @app.middleware("http")
        async def authenticate(request: Request, call_next):
            try:
                credentials: HTTPBasicCredentials = await security(request)
                if not (secrets.compare_digest(credentials.username, app_username)
                        and secrets.compare_digest(credentials.password, app_password)):
                    return Response(
                        "Unauthorized",
                        status_code=401,
                        headers={"WWW-Authenticate": "Basic"},
                    )
                return await call_next(request)
            except:
                return Response(
                    "Unauthorized",
                    status_code=401,
                    headers={"WWW-Authenticate": "Basic"},
                )

    @app.get("/sd-webui-prompt-all-in-one-js")
    async def sd_webui_prompt_all_in_one_js():
        js = ''
        for file in os.listdir(os.path.join(AIO_PATH, 'javascript')):
            if file.endswith('.js'):
                with open(os.path.join(AIO_PATH, 'javascript', file), 'r', encoding='utf-8') as f:
                    js += f.read() + '\n'
        response = Response(content=js, media_type="application/javascript")
        return response

    app_started_callback(Optional[Blocks], app)

    static_dir = os.path.join(BASE_DIR, "static")
    app.mount("/prompt-helper-static", StaticFiles(directory=static_dir, html=True), name="prompt-helper-static")

    print("")
    print(f"[Prompt Helper] Ready to mount at /prompt-helper/")
    return app


if __name__ == "__main__":
    # optional standalone mode for debugging
    import uvicorn
    app = create_prompt_helper_app()
    app_port = int(os.environ.get("APP_PORT", 17860))
    print(f"Listening on port {app_port}...")
    print(f"Open http://localhost:{app_port}/?__theme=dark to access this app.")
    uvicorn.run(app, host="0.0.0.0", port=app_port, log_level="warning")
