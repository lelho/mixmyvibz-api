"""MixMyVibz — serveur léger (Render gratuit, 512 Mo) : Auto-Tune + assistant IA. Pas de Demucs ici."""
import json, os, tempfile, uuid, urllib.request
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from autotune import autotune_long, detect_key_light

app = FastAPI(title="MixMyVibz API (léger)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
JOBS = {}
WORK = tempfile.mkdtemp(prefix="mixmyvibz_")
MAX_SECONDS = 420  # 7 min : au-delà, la mémoire de l'offre gratuite ne suffit plus
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@app.get("/api/health")
def health():
    return {"ok": True, "autotune": True, "separation": False, "master": False, "chat": bool(os.environ.get("MMV_LLM_KEY") or os.environ.get("ANTHROPIC_API_KEY"))}


@app.get("/")
def root():
    return {"service": "MixMyVibz API", "health": "/api/health"}


@app.get("/api/download/{name}")
def download(name: str):
    jid = name.replace(".wav", "")
    j = JOBS.get(jid)
    if not j or not os.path.exists(j["wav"]):
        raise HTTPException(404, "Fichier introuvable (expiré)")
    return FileResponse(j["wav"], media_type="audio/wav", filename=f"mixmyvibz_{jid}.wav")


@app.post("/api/autotune")
async def api_autotune(file: UploadFile = File(...), key: str = Form("auto"), mode: str = Form("minor"), speed: float = Form(30)):
    """Corrige la justesse d'une piste voix. key = 'auto' ou une note. speed en ms : 0 = effet robot, 80 = naturel."""
    ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    if ext not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        raise HTTPException(400, "Format non supporté")
    jid = uuid.uuid4().hex[:12]
    src = os.path.join(WORK, jid + ext)
    with open(src, "wb") as f:
        f.write(await file.read())
    try:
        info = sf.info(src) if ext in (".wav", ".flac", ".ogg") else None
        if info and info.duration > MAX_SECONDS:
            raise HTTPException(413, "Piste trop longue pour le serveur gratuit (7 min max)")
        y, sr = sf.read(src, dtype="float32", always_2d=True)
        mono = y.mean(axis=1).astype(np.float32)
        del y
        if len(mono) / sr > MAX_SECONDS:
            raise HTTPException(413, "Piste trop longue pour le serveur gratuit (7 min max)")
        if key == "auto":
            k = detect_key_light(mono, sr)
            key, mode = k["key"], k["mode"]
        if key not in NOTES or mode not in ("major", "minor"):
            raise HTTPException(400, "Tonalité invalide")
        tuned = autotune_long(mono, sr, key, mode, float(np.clip(speed, 0, 80)))
        out = os.path.join(WORK, jid + "_tuned.wav")
        sf.write(out, tuned, sr, subtype="PCM_24")
        JOBS[jid] = {"wav": out}
        if len(JOBS) > 20:
            JOBS.pop(next(iter(JOBS)), None)
        return {"id": jid, "wav": f"/api/download/{jid}.wav", "key": key, "mode": mode, "sr": int(sr)}
    finally:
        if os.path.exists(src):
            os.remove(src)


# ---------- Assistant IA ----------
CHAT_SYSTEM = (
    "Tu es Mastermind, le robot mascotte de MixMyVibz, un site de mixage, mastering et beatmaking assistés par IA. "
    "Tu as la culture des grands ingénieurs du son de ces vingt dernières années (Jaycen Joshua, Serban Ghenea, Manny Marroquin, Young Guru, Derek Ali, Nk.F, Leslie Brathwaite, Stuart White, Noah Shebib…) "
    "et ton cerveau s'appuie sur une base de 68 chaînes de mixage et mastering de hits (rap US et FR, afro, R&B, pop). "
    "Ces chaînes viennent d'un document généré par une IA : présente-les comme des points de départ à affiner à l'oreille, jamais comme les réglages exacts vérifiés d'un ingénieur. Ne prétends pas avoir mixé ou entendu ces morceaux. "
    "Tu réponds dans la langue de l'utilisateur (français, anglais ou espagnol), avec un ton chaleureux et direct, de façon concrète, à toute question technique : mixage, mastering, EQ, compression, reverb, delay, gain staging, LUFS, stéréo, voix, Auto-Tune, beatmaking, "
    "théorie musicale, audio numérique, matériel, DAW, et utilisation du site. Donne des valeurs de départ chiffrées quand c'est utile, "
    "explique pourquoi, reste concis (moins de 180 mots sauf demande de détail). Quand l'utilisateur vise un hit ou un style, propose la chaîne la plus proche et dis où l'appliquer sur le site (onglet Assistant du studio, « Appliquer cette chaîne »). "
    "Si une question est hors sujet technique, dis-le poliment et recentre."
)


class ChatIn(BaseModel):
    messages: list[dict]


@app.post("/api/chat")
def chat(body: ChatIn):
    llm_key, llm_url = os.environ.get("MMV_LLM_KEY"), os.environ.get("MMV_LLM_BASE_URL")
    if not llm_key or not llm_url:
        raise HTTPException(503, "Assistant IA non configuré sur ce serveur")
    msgs = [{"role": "assistant" if x.get("role") == "assistant" else "user", "content": str(x.get("content", ""))[:4000]} for x in body.messages[-12:]]
    if not msgs or msgs[0]["role"] != "user":
        raise HTTPException(400, "Message invalide")
    req = urllib.request.Request(
        llm_url.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": os.environ.get("MMV_LLM_MODEL", "glm-4.5-flash"), "max_tokens": 700, "messages": [{"role": "system", "content": CHAT_SYSTEM}] + msgs}).encode(),
        headers={"Authorization": "Bearer " + llm_key, "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            j = json.loads(r.read())
        return {"reply": j["choices"][0]["message"]["content"]}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, "Assistant indisponible : " + str(e)[:120])
