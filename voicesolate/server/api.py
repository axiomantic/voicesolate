import os
import sys
import re
import json
import shutil
import asyncio
import logging
import subprocess
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

from .job_manager import job_manager
from .engine_service import engine_service
from .pipeline_runner import run_scan_job, run_pipeline_job
from ..waveform import generate_macro_waveform_from_manifest, extract_peaks_from_wav
from ..script_parser import ScriptParser
from ..model_trainer import ModelTrainer

logger = logging.getLogger("voicesolate.api")

app = FastAPI(
    title="Voicesolate API",
    version="1.0.0",
    description="Decoupled Neural Vocal Isolation, STT Radar & Voice Synthesis Studio Backend"
)

# Enable CORS for local dev or remote browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    job_manager.set_event_loop(loop)
    logger.info("Voicesolate API backend initialized.")

# ----------------- MODELS -----------------

class ScanRequest(BaseModel):
    input_path: str
    script_path: Optional[str] = None
    provider: Optional[str] = None

class PipelineRequest(BaseModel):
    input_path: str
    characters: List[str]
    script_path: Optional[str] = None
    provider: Optional[str] = None
    min_duration: float = 3.0
    enhance: bool = True
    targets: List[str] = ["all"]
    no_train: bool = False
    no_aggregate: bool = False

class SynthesizeRequest(BaseModel):
    character_name: str
    episode_name: Optional[str] = None
    engine: str
    text: str
    speed: float = 1.0
    seed: int = 42
    ref_audio_path: Optional[str] = None

class BatchSynthesizeRequest(BaseModel):
    character_name: str
    episode_name: Optional[str] = None
    engines: List[str]
    text: str
    speed: float = 1.0
    seed: int = 42
    ref_audio_path: Optional[str] = None

class TrainModelRequest(BaseModel):
    character_name: str
    episode_name: Optional[str] = None
    engine: str

class ClearStepRequest(BaseModel):
    step: int
    episode_name: Optional[str] = None
    character_name: Optional[str] = None

class InstallEngineRequest(BaseModel):
    engine: str
    params: Optional[Dict[str, Any]] = None

# ----------------- SYSTEM ENDPOINTS -----------------

@app.get("/api/v1/system/status")
def get_system_status():
    return engine_service.get_system_status()

def _find_character_dir(character_name: Optional[str], episode: Optional[str] = None) -> Optional[Path]:
    if not character_name:
        return None
    out_root = Path("./output")
    if not out_root.exists():
        return None
    if episode:
        cand = out_root / episode / character_name
        if cand.exists():
            return cand
    for ep in out_root.iterdir():
        if ep.is_dir() and not ep.name.startswith(".") and (ep / character_name).exists():
            return ep / character_name
    return None

@app.get("/api/v1/system/engines")
def get_system_engines(character: Optional[str] = None, episode: Optional[str] = None):
    char_dir = _find_character_dir(character, episode)
    return engine_service.get_engines_status(char_dir)

@app.post("/api/v1/system/install_engine")
def install_engine(req: InstallEngineRequest, background_tasks: BackgroundTasks):
    job = job_manager.create_job("install_engine", {"engine": req.engine})
    
    def _do_install(job_id: str, engine_id: str):
        try:
            job_manager.update_job(job_id, progress=20.0, stage="installing", message=f"Configuring {engine_id} engine...")
            pkg_map = {
                "piper": "piper-tts",
                "piper-tts": "piper-tts",
                "xtts": "TTS",
                "xtts-v2": "TTS",
                "f5-tts": "f5-tts",
                "f5tts": "f5-tts"
            }
            pkg = pkg_map.get(engine_id.lower(), engine_id)
            proc = subprocess.run([sys.executable, "-m", "pip", "install", pkg], capture_output=True, text=True, timeout=180)
            if proc.returncode != 0:
                logger.error(f"pip install failed: {proc.stderr}")
                raise RuntimeError(proc.stderr[:200] or "pip install failed")
            import importlib
            importlib.invalidate_caches()
            job_manager.update_job(job_id, progress=100.0, stage="complete", status="completed", message=f"{engine_id} ({pkg}) installed successfully.")
        except Exception as e:
            job_manager.update_job(job_id, status="failed", error=str(e), message=f"Installation failed: {e}")

    background_tasks.add_task(_do_install, job.job_id, req.engine)
    return {"job_id": job.job_id, "status": "queued"}

# ----------------- SCRIPT DETECTION & UPLOAD -----------------

@app.get("/api/v1/scripts/detect")
def detect_script(filename: str):
    """
    Detects episode identifier (e.g. s06e01) from media filename and
    loads corresponding script & speaking characters.
    """
    fname = Path(filename).name
    m = re.search(r"s0?(\d+)[ex]0?(\d+)", fname, re.IGNORECASE)
    ep_code = None
    if m:
        s, e = int(m.group(1)), int(m.group(2))
        ep_code = f"s{s:02d}e{e:02d}"

    characters = []
    dialogues_count = 0
    provider = "startrek"
    
    if ep_code:
        try:
            parser = ScriptParser(use_cache=True)
            dialogues, _ = parser.fetch_or_load(ep_code, provider="startrek")
            dialogues_count = len(dialogues)
            for c in parser.get_characters_sorted():
                if c.line_count >= 1:
                    est_sec = c.line_count * 3.2
                    characters.append({
                        "name": c.name,
                        "lines": c.line_count,
                        "words": c.word_count,
                        "estimated_duration_min": round(est_sec / 60.0, 1)
                    })
        except Exception as e:
            logger.warning(f"Script detection fetch failed: {e}")

    return {
        "filename": fname,
        "detected_episode": ep_code,
        "provider": provider if ep_code else "generic",
        "dialogues_count": dialogues_count,
        "characters": characters
    }

@app.post("/api/v1/scripts/upload")
async def upload_script(file: UploadFile = File(...)):
    """
    Accepts uploaded script/subtitle files (.srt, .txt, .json),
    parses dialogue lines, and returns discovered character roster.
    """
    scripts_dir = Path("./cache/scripts").resolve()
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = scripts_dir / file.filename
    content = await file.read()
    with open(dest_path, "wb") as f:
        f.write(content)
        
    parser = ScriptParser(use_cache=False)
    dialogues, _ = parser.fetch_or_load(str(dest_path))
    characters = []
    for c in parser.get_characters_sorted():
        if c.line_count >= 1:
            est_sec = c.line_count * 3.2
            characters.append({
                "name": c.name,
                "lines": c.line_count,
                "words": c.word_count,
                "estimated_duration_min": round(est_sec / 60.0, 1)
            })

    return {
        "filename": file.filename,
        "saved_path": str(dest_path),
        "dialogues_count": len(dialogues),
        "characters": characters
    }

# ----------------- EPISODES & MANIFESTS -----------------

@app.get("/api/v1/episodes")
def list_episodes():
    """Scans output directory for processed or discovered episodes."""
    out_dir = Path("./output")
    if not out_dir.exists():
        return []

    episodes = []
    for d in sorted(out_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and not d.name.startswith(".") and d.name not in ("embedded_subs", "cache"):
            manifest_file = d / "manifest.json"
            char_dirs = [c.name for c in d.iterdir() if c.is_dir() and not c.name.startswith(".")]
            
            clips_count = 0
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        m = json.load(f)
                        clips_count = len(m.get("clips", []))
                except Exception:
                    pass

            if clips_count > 0 or char_dirs:
                episodes.append({
                    "id": d.name,
                    "name": d.name.replace("_", " "),
                    "path": str(d.resolve()),
                    "has_manifest": manifest_file.exists(),
                    "characters": char_dirs,
                    "clips_count": clips_count,
                    "modified_at": d.stat().st_mtime
                })
    return episodes

def _find_episode_dir(episode_name: str) -> Optional[Path]:
    for out_root in [Path("./output"), Path("./output2")]:
        if out_root.exists():
            cand = out_root / episode_name
            if cand.exists() and cand.is_dir():
                return cand
    return None

@app.get("/api/v1/episodes/{episode_name}")
def get_episode_details(episode_name: str):
    ep_dir = _find_episode_dir(episode_name)
    if not ep_dir:
        raise HTTPException(status_code=404, detail=f"Episode {episode_name} not found.")

    manifest_file = ep_dir / "manifest.json"
    manifest_data = {}
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    characters = []
    for c in ep_dir.iterdir():
        if c.is_dir() and not c.name.startswith("."):
            raw_clips = list((c / "raw").glob("*.wav")) if (c / "raw").exists() else []
            enh_clips = list((c / "enhanced").glob("*.wav")) if (c / "enhanced").exists() else []
            characters.append({
                "name": c.name,
                "raw_clips_count": len(raw_clips),
                "enhanced_clips_count": len(enh_clips),
                "has_datasets": (c / "datasets").exists(),
                "has_models": (c / "models").exists()
            })

    return {
        "episode_name": episode_name,
        "path": str(ep_dir.resolve()),
        "manifest": manifest_data,
        "characters": characters
    }

@app.get("/api/v1/episodes/{episode_name}/waveform")
def get_episode_waveform(episode_name: str):
    ep_dir = _find_episode_dir(episode_name)
    if not ep_dir:
        return {"points": [], "speech_spans": [], "duration": 0}
    manifest_file = ep_dir / "manifest.json"
    return generate_macro_waveform_from_manifest(manifest_file, num_points=1200)

# ----------------- CHARACTER STUDIO DETAILS -----------------

@app.get("/api/v1/characters/{character_name}/details")
def get_character_details(character_name: str, episode: Optional[str] = None):
    char_dir = _find_character_dir(character_name, episode)
    if not char_dir:
        raise HTTPException(status_code=404, detail=f"Character {character_name} not found in output.")

    quotes = engine_service.get_character_dialogue_quotes(char_dir)
    ref_prompts = engine_service.get_reference_prompts(char_dir)
    engines = engine_service.get_engines_status(char_dir)

    # Dataset stats
    piper_wavs = list((char_dir / "datasets" / "piper" / "wavs").glob("*.wav")) if (char_dir / "datasets" / "piper" / "wavs").exists() else []
    
    return {
        "character_name": character_name,
        "directory": str(char_dir.resolve()),
        "quotes": quotes,
        "reference_prompts": ref_prompts,
        "engines": engines,
        "dataset_stats": {
            "clip_count": len(piper_wavs),
            "piper_dataset_ready": (char_dir / "datasets" / "piper" / "metadata.csv").exists(),
            "xtts_dataset_ready": (char_dir / "datasets" / "xtts" / "metadata.csv").exists(),
            "f5tts_dataset_ready": (char_dir / "datasets" / "f5tts" / "metadata.csv").exists()
        }
    }

# ----------------- JOBS & BACKGROUND RUNNER -----------------

@app.post("/api/v1/pipeline/scan")
def scan_media(req: ScanRequest, background_tasks: BackgroundTasks):
    job = job_manager.create_job("scan", req.dict())
    background_tasks.add_task(run_scan_job, job.job_id, req.input_path, req.script_path, req.provider)
    return {"job_id": job.job_id, "status": "queued"}

@app.post("/api/v1/pipeline/run")
def run_pipeline(req: PipelineRequest, background_tasks: BackgroundTasks):
    job = job_manager.create_job("pipeline", req.dict())
    background_tasks.add_task(run_pipeline_job, job.job_id, req.dict())
    return {"job_id": job.job_id, "status": "queued"}

@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()

@app.post("/api/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or already terminated")
    return {"job_id": job_id, "status": "cancelled"}

# ----------------- SYNTHESIS -----------------

@app.post("/api/v1/synthesize")
def synthesize_speech(req: SynthesizeRequest):
    char_dir = _find_character_dir(req.character_name, req.episode_name)
    if not char_dir:
        raise HTTPException(status_code=404, detail=f"Character directory for '{req.character_name}' not found.")

    try:
        res = engine_service.synthesize(
            character_dir=char_dir,
            engine_id=req.engine,
            text=req.text,
            speed=req.speed,
            seed=req.seed,
            ref_audio_path=req.ref_audio_path
        )
        return res
    except Exception as e:
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/synthesize/batch")
def synthesize_batch(req: BatchSynthesizeRequest):
    """
    Synthesizes speech across multiple checked models concurrently/sequentially.
    """
    char_dir = _find_character_dir(req.character_name, req.episode_name)
    if not char_dir:
        raise HTTPException(status_code=404, detail=f"Character directory for '{req.character_name}' not found.")

    results = {}
    for eng in req.engines:
        try:
            res = engine_service.synthesize(
                character_dir=char_dir,
                engine_id=eng,
                text=req.text,
                speed=req.speed,
                seed=req.seed,
                ref_audio_path=req.ref_audio_path
            )
            results[eng] = {"status": "success", **res}
        except Exception as e:
            logger.warning(f"Synthesis error for {eng}: {e}")
            results[eng] = {"status": "error", "error": str(e)}

    return {
        "character": req.character_name,
        "text": req.text,
        "results": results
    }

# ----------------- TRAINING & MODEL COMPILATION -----------------

@app.post("/api/v1/training/train")
def train_model(req: TrainModelRequest, background_tasks: BackgroundTasks):
    """
    Triggers on-demand training or compilation for a specific model architecture.
    """
    char_dir = _find_character_dir(req.character_name, req.episode_name)
    if not char_dir:
        raise HTTPException(status_code=404, detail=f"Character directory for '{req.character_name}' not found.")

    job = job_manager.create_job("train", req.dict())

    def _do_train(job_id: str, cdir: Path, eng: str):
        try:
            job_manager.update_job(job_id, progress=10.0, stage="train", message=f"Initializing training for {eng.upper()} ({cdir.name})...")
            trainer = ModelTrainer(cdir)
            datasets = {
                "piper": cdir / "datasets" / "piper",
                "xtts": cdir / "datasets" / "xtts",
                "f5tts": cdir / "datasets" / "f5tts"
            }
            eng_norm = eng.lower().replace("-", "").replace("_", "")
            if "piper" in eng_norm or "onnx" in eng_norm:
                job_manager.update_job(job_id, progress=30.0, stage="train", message="Configuring Piper VITS & LJSpeech dataset...")
                res = trainer.train_piper(datasets["piper"])
            elif "xtts" in eng_norm:
                job_manager.update_job(job_id, progress=30.0, stage="train", message="Computing Coqui XTTS-v2 speaker conditioning latents...")
                res = trainer.train_xtts(datasets["xtts"])
            elif "f5" in eng_norm:
                job_manager.update_job(job_id, progress=30.0, stage="train", message="Configuring F5-TTS reference prompt pack & DiT profile...")
                res = trainer.train_f5tts(datasets["f5tts"])
            else:
                raise ValueError(f"Unknown engine: {eng}")

            job_manager.update_job(
                job_id,
                progress=100.0,
                stage="complete",
                status="completed",
                message=f"✓ {eng.upper()} voice model prepared successfully for {cdir.name}!",
                result={"model_dir": str(res)} if res else {}
            )
        except Exception as e:
            logger.exception("Training failed")
            job_manager.update_job(job_id, status="failed", error=str(e), message=f"Training error: {e}")

    background_tasks.add_task(_do_train, job.job_id, char_dir, req.engine)
    return {"job_id": job.job_id, "status": "queued"}

# ----------------- WIZARD STEP CLEARING -----------------

@app.post("/api/v1/steps/clear")
def clear_step(req: ClearStepRequest):
    """
    Clears state and cache files corresponding to a specific wizard step.
    """
    step = req.step
    cleared = []

    if step == 1:
        return {"status": "cleared", "step": 1, "details": "Reset step 1 selections"}

    elif step == 2:
        char_dir = _find_character_dir(req.character_name, req.episode_name)
        if char_dir:
            raw_dir = char_dir / "raw"
            if raw_dir.exists():
                shutil.rmtree(raw_dir, ignore_errors=True)
                cleared.append(str(raw_dir))
            enh_dir = char_dir / "enhanced"
            if enh_dir.exists():
                shutil.rmtree(enh_dir, ignore_errors=True)
                cleared.append(str(enh_dir))

        ep_dir = _find_episode_dir(req.episode_name) if req.episode_name else None
        if ep_dir:
            manifest_file = ep_dir / "manifest.json"
            if manifest_file.exists():
                manifest_file.unlink(missing_ok=True)
                cleared.append(str(manifest_file))

        return {"status": "cleared", "step": 2, "cleared": cleared}

    elif step == 3:
        char_dir = _find_character_dir(req.character_name, req.episode_name)
        if char_dir:
            ds_dir = char_dir / "datasets"
            if ds_dir.exists():
                shutil.rmtree(ds_dir, ignore_errors=True)
                cleared.append(str(ds_dir))
            models_dir = char_dir / "models"
            if models_dir.exists():
                shutil.rmtree(models_dir, ignore_errors=True)
                cleared.append(str(models_dir))
        return {"status": "cleared", "step": 3, "cleared": cleared}

    elif step == 4:
        cache_dir = Path("cache/synthesized").resolve()
        if cache_dir.exists():
            for f in cache_dir.glob("*.wav"):
                if not req.character_name or req.character_name.upper() in f.name.upper():
                    f.unlink(missing_ok=True)
                    cleared.append(str(f))
        return {"status": "cleared", "step": 4, "cleared": cleared}

    else:
        raise HTTPException(status_code=400, detail=f"Invalid step: {step}")

# ----------------- AUDIO STREAMING -----------------

@app.get("/api/v1/audio/stream")
@app.head("/api/v1/audio/stream")
def stream_audio(path: str = Query(..., description="Absolute path to audio file")):
    audio_path = Path(path).resolve()
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {path}")

    # MIME type
    ext = audio_path.suffix.lower()
    media_type = "audio/wav" if ext == ".wav" else ("audio/mpeg" if ext == ".mp3" else "application/octet-stream")
    return FileResponse(
        str(audio_path),
        media_type=media_type,
        filename=audio_path.name
    )

# ----------------- WEBSOCKET -----------------

@app.websocket("/ws/pipeline")
async def websocket_pipeline_endpoint(websocket: WebSocket):
    await job_manager.connect_socket(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Respond to ping or client commands
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
            except Exception:
                pass
    except WebSocketDisconnect:
        job_manager.disconnect_socket(websocket)
    except Exception:
        job_manager.disconnect_socket(websocket)

# ----------------- STATIC WEB UI -----------------

web_dir = Path(__file__).parent.parent / "web"
static_dir = web_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
@app.head("/")
def serve_index():
    index_file = web_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="""
    <html>
      <head><title>Voicesolate Studio</title></head>
      <body style="background:#0f172a;color:#fff;font-family:sans-serif;padding:40px;">
        <h2>Voicesolate API Backend Online</h2>
        <p>Frontend static assets loading...</p>
      </body>
    </html>
    """)
