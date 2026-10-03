"""ClipCut pipeline: download -> transcribe -> score -> cut -> caption -> render."""
import json
import os
import re
import subprocess
import threading
import wave as wv
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")  # tiny/base/small/medium
MAX_VIDEO_SECONDS = 3 * 3600

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        return _model


def run(cmd, timeout=None, cwd=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-600:]}")
    return p


YOUTUBE_RE = re.compile(
    r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch\?[^ ]*v=|shorts/)|youtu\.be/)[\w\-]{6,}"
)


def validate_url(url: str) -> str:
    url = (url or "").strip()
    if not YOUTUBE_RE.match(url):
        raise ValueError("Ye valid YouTube video link nahi lagta.")
    return url


def resolve_info(url: str) -> dict:
    p = run(["yt-dlp", "--dump-single-json", "--skip-download", "--no-playlist",
             "--no-warnings", url], timeout=45)
    info = json.loads(p.stdout)
    return {"title": info.get("title"), "duration": info.get("duration"),
            "thumbnail": info.get("thumbnail"), "uploader": info.get("uploader")}


def download_video(url: str, workdir: Path) -> Path:
    """Try default web client, fall back to android/ios player clients
    (datacenter IPs often get 429 on the default client)."""
    out = workdir / "video.%(ext)s"
    last_err = None
    for client in (None, "android", "ios"):
        args = ["yt-dlp", "-f", "bv*[height<=720]+ba/b[height<=720]/b",
                "--merge-output-format", "mp4", "--no-playlist",
                "--no-warnings", "-o", str(out)]
        if client:
            args += ["--extractor-args", f"youtube:player_client={client}"]
        args.append(url)
        try:
            run(args, timeout=1200)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    else:
        raise RuntimeError(f"YouTube ne download block kar diya: {str(last_err)[-200:]}")
    for ext in ("mp4", "mkv", "webm"):
        cand = workdir / f"video.{ext}"
        if cand.exists():
            if ext != "mp4":
                run(["ffmpeg", "-y", "-i", str(cand), "-c", "copy",
                     str(workdir / "video.mp4")], timeout=600)
                cand.unlink()
            return workdir / "video.mp4"
    raise RuntimeError("Download ho gaya lekin video file nahi mili.")


def extract_audio(video: Path, workdir: Path) -> Path:
    wav = workdir / "audio.wav"
    run(["ffmpeg", "-y", "-i", str(video), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", str(wav)], timeout=600)
    return wav


def transcribe(wav: Path):
    model = get_model()
    segments, _ = model.transcribe(str(wav), word_timestamps=True)
    words = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            t = w.word.strip()
            if t:
                words.append({"w": t, "start": w.start, "end": w.end})
    return words


def sentences_from_words(words):
    sents, cur = [], []
    for wd in words:
        cur.append(wd)
        if re.search(r"[.?!…]['\"]?$", wd["w"]):
            sents.append({"text": " ".join(x["w"] for x in cur),
                          "start": cur[0]["start"], "end": cur[-1]["end"]})
            cur = []
    if cur:
        sents.append({"text": " ".join(x["w"] for x in cur),
                      "start": cur[0]["start"], "end": cur[-1]["end"]})
    return sents


def energy_profile(wav: Path):
    with wv.open(str(wav), "rb") as f:
        raw = f.readframes(f.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    sr, secs = 16000, max(1, len(arr) // 16000)
    prof = np.zeros(secs)
    for i in range(secs):
        chunk = arr[i * sr:(i + 1) * sr]
        prof[i] = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
    mx = prof.max()
    return prof / mx if mx > 0 else prof


HOOK_RES = [
    (re.compile(r"\b(how to|how do|why|what if|secret|mistake|stop doing|never|nobody|truth|exposed|free|warning|imagine)\b", re.I), 12),
    (re.compile(r"\?"), 8),
    (re.compile(r"\b\d+\b"), 6),
]
KEYWORD_RES = [
    re.compile(r"\b(money|success|rich|million|billion|profit|business)\b", re.I),
    re.compile(r"\b(love|heart|mother|father|family|cry|tears|sad|happy)\b", re.I),
    re.compile(r"\b(fact|amazing|incredible|unbelievable|shocking|crazy)\b", re.I),
    re.compile(r"\b(danger|kill|death|fight|war|accident|scary)\b", re.I),
    re.compile(r"\b(health|doctor|disease|weight|sleep|brain)\b", re.I),
]
PROMO_RE = re.compile(r"\b(subscribe|like this video|link in (the )?description|sponsor|use code|discount)\b", re.I)
FILLER_RE = re.compile(r"^(um+|uh+|like|you know|basically|actually)$", re.I)


def score_window(text_low, first_low, n_words, dur, energy, pauses, filler_ratio):
    score = 0.0
    for rx, pts in HOOK_RES:
        if rx.search(first_low):
            score += pts
    kw_hits = sum(1 for rx in KEYWORD_RES if rx.search(text_low))
    score += min(kw_hits * 6, 24)
    score += min(energy * 16, 16)
    wps = n_words / max(dur, 1)
    score += 10 if 2.0 <= wps <= 3.4 else (6 if 1.2 <= wps < 2.0 else 2)
    if pauses:
        score -= 12
    score -= min(filler_ratio * 40, 10)
    if PROMO_RE.search(text_low):
        score -= 30
    return max(0.0, min(100.0, score))


def pick_windows(words, sents, energy, total_dur, clip_dur, num_clips):
    if not words:
        return []
    step = max(5, clip_dur // 6)
    cands, t = [], 0.0
    while t + clip_dur * 0.6 <= total_dur:
        s_start, best = t, None
        for s in sents:
            if s["start"] < t - 6:
                continue
            if s["start"] > t + 3:
                break
            d = abs(s["start"] - t)
            if best is None or d < best[0]:
                best = (d, s["start"])
        if best:
            s_start = best[1]
        target_end, s_end, best = s_start + clip_dur, s_start + clip_dur, None
        for s in sents:
            if s["end"] < target_end - 8:
                continue
            if s["end"] > target_end + 8:
                break
            d = abs(s["end"] - target_end)
            if best is None or d < best[0]:
                best = (d, s["end"])
        if best:
            s_end = best[1]
        dur = s_end - s_start
        if dur < clip_dur * 0.55 or dur > clip_dur * 1.5:
            t += step
            continue
        win_words = [w for w in words
                     if w["start"] >= s_start - 0.05 and w["end"] <= s_end + 0.05]
        if len(win_words) < 8:
            t += step
            continue
        text = " ".join(w["w"] for w in win_words)
        text_low = text.lower()
        first_low = " ".join(w["w"] for w in win_words[:12]).lower()
        e0, e1 = int(s_start), min(len(energy), int(s_end) + 1)
        e = float(np.mean(energy[e0:e1])) if e1 > e0 else 0.0
        pauses = any(win_words[i + 1]["start"] - win_words[i]["end"] > 1.6
                     for i in range(len(win_words) - 1))
        fillers = sum(1 for w in win_words if FILLER_RE.match(w["w"].lower()))
        sc = score_window(text_low, first_low, len(win_words), dur, e, pauses,
                          fillers / max(len(win_words), 1))
        cands.append({"start": round(s_start, 2), "end": round(s_end, 2),
                      "dur": round(dur, 2), "score": round(sc, 1),
                      "text": text, "words": win_words})
        t += step
    cands.sort(key=lambda c: c["score"], reverse=True)
    chosen = []
    for c in cands:
        if len(chosen) >= num_clips:
            break
        if all(c["start"] >= o["end"] + 1 or c["end"] <= o["start"] - 1 for o in chosen):
            chosen.append(c)
    return chosen


def _ass_time(sec):
    sec = max(0, sec)
    return f"{int(sec // 3600)}:{int((sec % 3600) // 60):02d}:{(sec % 60):05.2f}"


def _esc(t):
    return t.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def build_ass(clip_words, clip_start, style, width, height, path: Path):
    narrow = width <= 720
    fontsize = 56 if narrow else 48
    margin_v = 150 if height > width else 70
    max_words = 3 if narrow else 6  # short lines so text never overflows frame
    lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}",
             f"PlayResY: {height}", "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
             "[V4+ Styles]",
             "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
             "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
             "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
             "MarginL, MarginR, MarginV, Encoding",
             f"Style: Cap,DejaVu Sans,{fontsize},&H00FFFFFF,&H000019FF,&H00000000,"
             f"&H96000000,-1,0,0,0,100,100,0,0,1,4,1,2,40,40,{margin_v},1",
             "", "[Events]",
             "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    groups, cur = [], []
    for w in clip_words:
        cur.append(w)
        if len(cur) >= max_words or re.search(r"[.?!…]['\"]?$", w["w"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    for i, g in enumerate(groups):
        s = g[0]["start"] - clip_start
        e = g[-1]["end"] - clip_start + 0.25
        if i + 1 < len(groups):
            # never overlap the next caption line
            e = min(e, groups[i + 1][0]["start"] - clip_start - 0.01)
            e = max(e, s + 0.3)
        if style == "karaoke":
            body = "".join("{\\k%d}%s " % (max(1, int((w["end"] - w["start"]) * 100)),
                                           _esc(w["w"])) for w in g).strip()
        else:
            body = _esc(" ".join(w["w"] for w in g))
        lines.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Cap,,0,0,0,,{body}")
    path.write_text("\n".join(lines), encoding="utf-8")


def render_clip(video: Path, workdir: Path, idx: int, win: dict, aspect: str,
                captions: bool, cap_style: str):
    name = f"clip_{idx + 1}"
    if aspect == "9:16":
        w, h = 720, 1280
        scale_crop = "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280"
    elif aspect == "1:1":
        w, h = 720, 720
        scale_crop = "scale=720:720:force_original_aspect_ratio=increase,crop=720:720"
    else:
        w, h = 1280, 720
        scale_crop = ("scale=1280:720:force_original_aspect_ratio=decrease,"
                      "pad=1280:720:(ow-iw)/2:(oh-ih)/2")
    vf = scale_crop
    ass = workdir / f"{name}.ass"
    if captions:
        build_ass(win["words"], win["start"], cap_style, w, h, ass)
        vf += f",subtitles={ass.name}"
    out = workdir / f"{name}.mp4"
    run(["ffmpeg", "-y", "-i", str(video), "-ss", str(win["start"]), "-t", str(win["dur"]),
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)],
        timeout=900, cwd=str(workdir))
    thumb = workdir / f"{name}.jpg"
    run(["ffmpeg", "-y", "-ss", str(win["dur"] / 2), "-i", str(out),
         "-frames:v", "1", "-q:v", "4", str(thumb)], timeout=120, cwd=str(workdir))
    if ass.exists():
        ass.unlink()
    return out.name, thumb.name


def run_pipeline(job: dict, set_progress, fail):
    """job: dict with id/url/settings. set_progress(step, pct, msg), fail(msg)."""
    job_id = job["id"]
    s = job["settings"]
    workdir = JOBS_DIR / job_id
    workdir.mkdir(exist_ok=True)
    try:
        if job.get("local_video") and (workdir / "video.mp4").exists():
            set_progress("upload", 5, "Video mil gayi…")
            video = workdir / "video.mp4"
        else:
            set_progress("downloading", 5, "Video download ho rahi hai…")
            video = download_video(job["url"], workdir)

        set_progress("transcribing", 22, "Awaz ko text me badla ja raha hai…")
        wav = extract_audio(video, workdir)
        words = transcribe(wav)
        if len(words) < 20:
            raise RuntimeError("Video me qaabil-e-istamal speech nahi mili.")
        total_dur = words[-1]["end"]
        if total_dur > MAX_VIDEO_SECONDS:
            raise RuntimeError("Video 3 ghante se lambi hai.")

        set_progress("analyzing", 58, "Best moments dhoonde ja rahe hain…")
        sents = sentences_from_words(words)
        energy = energy_profile(wav)
        wins = pick_windows(words, sents, energy, total_dur,
                            s["clip_duration"], s["num_clips"])
        if not wins:
            raise RuntimeError("Koi munasib clip nahi mil saki.")

        clips = []
        for i, win in enumerate(wins):
            pct = 65 + int(30 * (i + 1) / len(wins))
            set_progress("rendering", pct, f"Clip {i + 1}/{len(wins)} ban rahi hai…")
            mp4, jpg = render_clip(video, workdir, i, win, s["aspect_ratio"],
                                   s["captions"], s["caption_style"])
            clips.append({"file": mp4, "thumbnail": jpg, "score": win["score"],
                          "duration": win["dur"], "start": win["start"],
                          "preview_text": win["text"][:140]})
        for p in ("video.mp4", "audio.wav"):
            f = workdir / p
            if f.exists():
                f.unlink()
        job.update(status="done", step="done", percent=100, message="",
                   clips=clips, error=None)
    except Exception as e:
        fail(str(e)[:500])
