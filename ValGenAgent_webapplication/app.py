import os
import pty
import select
import subprocess
import threading
import asyncio
import time
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import uvicorn

OUTPUT_DIR = "/software/users/rajprinc/test_results_dir/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
first_time_setup_done = False
stop_flag = False

uploaded_feature_file = None
uploaded_code_zip = None
uploaded_docs_zip = None
uploaded_url_file=None

@app.get("/")
async def get_index():
    with open("templates/index.html") as f:
        return HTMLResponse(f.read())


@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    global uploaded_feature_file
    save_path = os.path.join(OUTPUT_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    uploaded_feature_file = save_path
    return JSONResponse({"status": "ok", "path": save_path})


@app.post("/upload_folder")
async def upload_folder(file: UploadFile = File(...), folder_type: str = Form(...)):
    global uploaded_code_zip, uploaded_docs_zip
    save_path = os.path.join(OUTPUT_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    if folder_type == "code":
        uploaded_code_zip = save_path
    elif folder_type == "docs":
        uploaded_docs_zip = save_path
    print(uploaded_docs_zip)
    print(uploaded_code_zip)

    return JSONResponse({"status": "ok", "folder_type": folder_type, "path": save_path})


@app.post("/upload_urls_file")
async def upload_urls_file(file: UploadFile = File(...)):
    # input_dir = os.path.join("ValGenAgent", "input_dirs")
    # os.makedirs(input_dir, exist_ok=True)
    # save_path = os.path.join(input_dir, "public_urls.txt")
    # with open(save_path, "wb") as f:
    #     f.write(await file.read())
    # return JSONResponse({"status": "ok", "path": save_path})
    global uploaded_url_file
    save_path = os.path.join(OUTPUT_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    uploaded_url_file = save_path
    return JSONResponse({"status": "ok", "path": save_path})



@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global first_time_setup_done, stop_flag
    await ws.accept()

    mode = await ws.receive_text()           # connect/run
    connection_type = await ws.receive_text()  # container/custom
    value = await ws.receive_text()          # container name or custom command
    functionality = await ws.receive_text()  # selected functionality

    master_fd, slave_fd = pty.openpty()

    # Launch based on selection
    if connection_type == "container":
        proc = subprocess.Popen(
            ["hlctl", "container", "exec", "-w", value, "bash", "-n", "qa"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=True,
            bufsize=0,
        )
    else:
        proc = subprocess.Popen(
            value,
            shell=True,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=True,
            bufsize=0,
        )

    stop_flag = False

    def read_from_proc():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while not stop_flag:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in r:
                try:
                    data = os.read(master_fd, 1024).decode(errors="ignore")
                    if data:
                        try:
                            loop.run_until_complete(ws.send_text(data))
                        except RuntimeError:
                            break
                except OSError:
                    break
            if proc.poll() is not None:
                break

    threading.Thread(target=read_from_proc, daemon=True).start()

    if mode == "connect":
        if not first_time_setup_done:
            setup_cmds = [
                "apt-get install -y zip",
                "git clone https://github.com/mansi05ag/ValGenAgent.git",
                "cd ValGenAgent",
                "pip install -r requirements.txt",
                "exit"
            ]
            command = '\n'.join(setup_cmds) + '\n'
            os.write(master_fd, command.encode())

            while proc.poll() is None:
                await asyncio.sleep(0.2)

            first_time_setup_done = True

        await ws.send_text("[INFO] Connected and setup complete.")

    elif mode == "run":
        print()
        timestamp = str(int(time.time()))
        zip_name = f"test_results_{timestamp}.zip"
        zip_path = os.path.join(OUTPUT_DIR, zip_name)

        commands = "cd ValGenAgent/\n"

        commands += f"rm -rf input_dirs/code && unzip -o {uploaded_code_zip} -d input_dirs/code\n"
        commands += f"rm -rf input_dirs/docs && unzip -o {uploaded_docs_zip} -d input_dirs/docs\n"
        commands += f"rm -rf input_dirs/public_urls.txt && cp {uploaded_url_file} input_dirs/\n"
        feature_file_path = uploaded_feature_file if uploaded_feature_file else "input.txt"

        if functionality == "generate_and_execute":
            commands += f"PYTHONUNBUFFERED=1 python agents/test_codegen_agent.py --test-plan {feature_file_path} --output-dir test_output_final\n"
        elif functionality == "complete_workflow":
            commands += f"PYTHONUNBUFFERED=1 python test_runner.py --feature-input {feature_file_path} --output-dir test_results\n"
        elif functionality == "generate_plan_only":
            commands += f"PYTHONUNBUFFERED=1 python test_runner.py --feature-input {feature_file_path} --generate-plan-only --output-dir test_results\n"
        elif functionality == "run_test_automation":
            commands += f"PYTHONUNBUFFERED=1 python test_runner.py --test-automation-only --test-plan {feature_file_path} --output-dir test_results\n"
        elif functionality == "generate_tests_from_plan":
            commands += f"python test_runner.py --test-plan {feature_file_path} --output-dir output --execute-tests=false\n"

        elif functionality == "generate_tests_from_raw":
            commands += f"python test_runner.py --feature-input {uploaded_feature_file} --output-dir output --execute-tests=false\n"

        else:
            await ws.send_text("[ERROR] Invalid functionality selected.")
            await ws.close()
            return

        commands += (
            f"zip -r {zip_name} test_results\n"
            f"cp {zip_name} {zip_path}\n"
            "rm -r test_results\n"
            "exit\n"
        )

        os.write(master_fd, commands.encode())

        while proc.poll() is None:
            await asyncio.sleep(0.2)

        stop_flag = True
        await ws.send_text(f"[DONE] /download/{zip_name}")
        await ws.close()


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    return FileResponse(file_path, filename=filename, media_type="application/zip")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
