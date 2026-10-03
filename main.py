"""ClipCut backend API."""
import queue
import shutil
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import pipeline as P

ASPECTS = {"9:16", "1:1", "16:9"}

app = FastAPI(title="ClipCut Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
app.mount("/files", StaticFiles(directory=str(P.JOBS_DIR)), name="files")

jobs: dict[str, dict] = {}
job_queue: "queue.Queue[str]" = queue.Queue()
JOB_TTL = 24 * 3600


def set_progress(job_id, step, percent, message=""):
    j = jobs.get(job_id)
    if j:
        j.update(status="processing", step=step, percent=percent, message=message)


def fail(job_id, message):
    j = jobs.get(job_id)
    if j:
        j.update(status="failed", step="failed", percent=100, message="",
                 error=str(message)[:500])


def _worker():
    while True:
        job_id = job_queue.get()
        job = jobs.get(job_id)
        if job:
            try:
                P.run_pipeline(job,
                               lambda s, p, m: set_progress(job_id, s, p, m),
                               lambda m: fail(job_id, m))
            except Exception as e:  # noqa: BLE001
                fail(job_id, e)
        now = time.time()
        for jid, j in list(jobs.items()):
            if now - j.get("created", now) > JOB_TTL:
                shutil.rmtree(P.JOBS_DIR / jid, ignore_errors=True)
                jobs.pop(jid, None)
        job_queue.task_done()


threading.Thread(target=_worker, daemon=True).start()


@app.get("/")
def health():
    return {"ok": True, "service": "clipcut", "model": P.WHISPER_MODEL,
            "queued": job_queue.qsize()}


@app.get("/resolve")
def resolve(url: str):
    try:
        url = P.validate_url(url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        return P.resolve_info(url)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Video info nahi mil saki: {str(e)[:200]}")


def _make_job(url, title, clip_duration, num_clips, aspect, captions,
              cap_style, local_video=False):
    job_id = uuid.uuid4().hex[:12]
    (P.JOBS_DIR / job_id).mkdir(exist_ok=True)
    jobs[job_id] = {
        "id": job_id, "status": "queued", "step": "queued", "percent": 0,
        "message": "Queue me hai…", "error": None, "clips": [],
        "created": time.time(), "title": title or "", "url": url,
        "local_video": local_video,
        "settings": {"clip_duration": clip_duration, "num_clips": num_clips,
                     "aspect_ratio": aspect, "captions": captions,
                     "caption_style": cap_style},
    }
    job_queue.put(job_id)
    return {"job_id": job_id}


def _validate_settings(clip_duration, num_clips, aspect, cap_style):
    if aspect not in ASPECTS:
        raise HTTPException(400, "Aspect ratio 9:16, 1:1 ya 16:9 ho.")
    if not 5 <= clip_duration <= 90:
        raise HTTPException(400, "Clip duration 5–90 seconds ke darmiyan ho.")
    if not 1 <= num_clips <= 10:
        raise HTTPException(400, "Clips 1–10 tak ho sakte hain.")
    if cap_style not in ("karaoke", "plain"):
        cap_style = "karaoke"
    return cap_style


@app.post("/jobs")
def create_job(body: dict):
    try:
        url = P.validate_url(body.get("url", ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        clip_duration = int(body.get("clip_duration", 30))
        num_clips = int(body.get("num_clips", 3))
    except (TypeError, ValueError):
        raise HTTPException(400, "Duration aur clips ki tadaad numbers me do.")
    cap_style = _validate_settings(clip_duration, num_clips,
                                   body.get("aspect_ratio", "9:16"),
                                   body.get("caption_style", "karaoke"))
    return _make_job(url, body.get("title", ""), clip_duration, num_clips,
                     body.get("aspect_ratio", "9:16"),
                     bool(body.get("captions", True)), cap_style)


@app.post("/jobs/upload")
async def create_upload_job(
    file: UploadFile = File(...),
    clip_duration: int = Form(30),
    num_clips: int = Form(3),
    aspect_ratio: str = Form("9:16"),
    captions: bool = Form(True),
    caption_style: str = Form("karaoke"),
    title: str = Form(""),
):
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ("mp4", "mov", "mkv", "webm", "3gp", "m4v"):
        raise HTTPException(400, "Video file mp4/mov/mkv/webm me ho.")
    cap_style = _validate_settings(clip_duration, num_clips, aspect_ratio,
                                   caption_style)
    job_id = uuid.uuid4().hex[:12]
    workdir = P.JOBS_DIR / job_id
    workdir.mkdir(exist_ok=True)
    tmp = workdir / f"upload.{ext}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        P.run(["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
               str(workdir / "video.mp4")], timeout=900)
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(400, f"Video file samajh nahi ayi: {str(e)[:150]}")
    finally:
        tmp.unlink(missing_ok=True)
    jobs[job_id] = {
        "id": job_id, "status": "queued", "step": "queued", "percent": 0,
        "message": "Queue me hai…", "error": None, "clips": [],
        "created": time.time(), "title": title or file.filename or "",
        "url": "", "local_video": True,
        "settings": {"clip_duration": clip_duration, "num_clips": num_clips,
                     "aspect_ratio": aspect_ratio, "captions": captions,
                     "caption_style": cap_style},
    }
    job_queue.put(job_id)
    return {"job_id": job_id}


def _view(j: dict, base: str):
    out = {k: v for k, v in j.items() if k != "url"}
    if j.get("status") == "done":
        out["clips"] = [
            {**c, "url": f"{base}files/{j['id']}/{c['file']}",
             "thumbnail_url": f"{base}files/{j['id']}/{c['thumbnail']}"}
            for c in j["clips"]
        ]
    return out


@app.get("/jobs")
def list_jobs(request: Request):
    base = str(request.base_url)
    return {"jobs": [
        {"id": j["id"], "status": j["status"], "step": j["step"],
         "percent": j["percent"], "created": j["created"],
         "title": j.get("title", ""), "settings": j["settings"],
         "clip_count": len(j.get("clips", [])),
         "thumbnail_url": (f"{base}files/{j['id']}/{j['clips'][0]['thumbnail']}"
                           if j.get("clips") else None)}
        for j in sorted(jobs.values(), key=lambda x: x["created"], reverse=True)[:50]
    ]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job nahi mili.")
    return _view(j, str(request.base_url))


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    j = jobs.pop(job_id, None)
    if not j:
        raise HTTPException(404, "Job nahi mili.")
    shutil.rmtree(P.JOBS_DIR / job_id, ignore_errors=True)
    return {"ok": True}
