"""MixMyVibz — serveur léger (Render gratuit, 512 Mo) : Auto-Tune + assistant IA. Pas de Demucs ici."""
import json, os, tempfile, uuid, urllib.request, urllib.error
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
    return {"ok": True, "autotune": True, "separation": False, "master": False, "chat": bool(os.environ.get("MMV_LLM_KEY") or os.environ.get("ANTHROPIC_API_KEY")), "payments": bool(os.environ.get("STRIPE_SECRET_KEY"))}


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
    "et ton cerveau s'appuie sur une base de 69 chaînes de mixage et mastering de hits (rap US et FR, afro, R&B, pop). "
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




# ---- Paiement Stripe (Checkout hébergé : le site ne touche jamais aux cartes) ----
import hmac, time, urllib.parse
from fastapi import Header, Request

SITE_URL = os.environ.get("MMV_SITE_URL", "https://mixmyvibz.com").rstrip("/")
# Identifiants de prix (non secrets). Mode test pour l'instant ; on les remplacera par ceux du mode réel.
PRICES = {
    "single": os.environ.get("STRIPE_PRICE_SINGLE", "price_1UPym3Eh5G6wHLM8sEOfc4lA"),
    "pro": os.environ.get("STRIPE_PRICE_PRO", "price_1UPymqEh5G6wHLM8ib8uwraT"),
    "studio": os.environ.get("STRIPE_PRICE_STUDIO", "price_1UPynaEh5G6wHLM89qz4kv8R"),
}
PLAN_MODE = {"single": "payment", "pro": "subscription", "studio": "subscription"}
LIMITS = {
    "single": {"mixes": 1, "maxTracks": 8, "beatmaker": False},
    "pro": {"mixes": 15, "maxTracks": 20, "beatmaker": True},
    "studio": {"mixes": None, "maxTracks": 50, "beatmaker": True},
}


def _stripe(method, path, form=None):
    sk = os.environ.get("STRIPE_SECRET_KEY", "")
    if not sk:
        raise HTTPException(503, "Paiement non configuré sur ce serveur")
    data = urllib.parse.urlencode(form).encode() if form else None
    req = urllib.request.Request("https://api.stripe.com/v1/" + path, data=data, method=method, headers={"Authorization": "Bearer " + sk, **({"content-type": "application/x-www-form-urlencoded"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")[:300].replace(sk, "***")
        raise HTTPException(502 if e.code >= 500 else 400, "Stripe : " + msg)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, "Stripe injoignable : " + str(e)[:120].replace(sk, "***"))


_FAILS = {}  # ip -> [horodatages des échecs récents]
MAX_FAILS, FAIL_WINDOW = 8, 900  # 8 mauvaises clés en 15 min -> blocage temporaire


def _ip(request):
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?"))


def _is_admin(key, request=None):
    expected = os.environ.get("MMV_ADMIN_KEY", "")
    if not key:
        return False
    ip, now = (_ip(request) if request else "?"), time.time()
    recent = [t for t in _FAILS.get(ip, []) if now - t < FAIL_WINDOW]
    if len(recent) >= MAX_FAILS:
        _FAILS[ip] = recent
        raise HTTPException(429, "Trop d'essais. Réessaie dans 15 minutes.")
    ok = bool(expected) and hmac.compare_digest(str(key).encode(), expected.encode())
    if not ok:
        recent.append(now)
    _FAILS[ip] = recent
    return ok


class CheckoutIn(BaseModel):
    plan: str


@app.post("/api/checkout")
def checkout(body: CheckoutIn):
    plan = body.plan
    if plan not in PRICES:
        raise HTTPException(400, "Offre inconnue")
    form = {
        "mode": PLAN_MODE[plan],
        "line_items[0][price]": PRICES[plan],
        "line_items[0][quantity]": "1",
        "success_url": SITE_URL + "/compte.html?paid=1&session_id={CHECKOUT_SESSION_ID}",
        "cancel_url": SITE_URL + "/pricing.html?cancelled=1",
        "metadata[site]": "mixmyvibz",
        "metadata[plan]": plan,
    }
    if PLAN_MODE[plan] == "subscription":
        form["subscription_data[metadata][site]"] = "mixmyvibz"
        form["subscription_data[metadata][plan]"] = plan
    s = _stripe("POST", "checkout/sessions", form)
    return {"url": s["url"]}


def _entitlement_from_session(sid):
    if not sid.startswith("cs_") or len(sid) > 200 or not sid.replace("_", "").isalnum():
        raise HTTPException(400, "Code invalide")
    s = _stripe("GET", "checkout/sessions/" + sid + "?expand[]=subscription")
    md = s.get("metadata") or {}
    plan = md.get("plan")
    if md.get("site") != "mixmyvibz" or plan not in PLAN_MODE:
        raise HTTPException(404, "Code inconnu")
    if PLAN_MODE[plan] == "subscription":
        sub = s.get("subscription") or {}
        active = isinstance(sub, dict) and sub.get("status") in ("active", "trialing")
    else:
        active = s.get("payment_status") == "paid"
    return {"active": bool(active), "plan": plan, "limits": LIMITS[plan], "recurring": PLAN_MODE[plan] == "subscription", "email": ((s.get("customer_details") or {}).get("email") or "")}


# ---- Accès invités : codes signés (pas de base de données) ----
# Un code = offre + date de fin + identifiant + libellé, signé avec la clé admin. Changer la clé admin invalide tous les codes.
# Pour retirer un seul invité : ajouter son identifiant (affiché à la création) dans la variable MMV_REVOKED_GUESTS, séparés par des virgules.
import base64, time as _time, secrets


def _b64(b):
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload):
    return _b64(hmac.new(os.environ.get("MMV_ADMIN_KEY", "").encode(), payload.encode(), "sha256").digest()[:18])


class GuestIn(BaseModel):
    plan: str = "pro"
    days: int = 7
    label: str = ""


@app.post("/api/admin/guest")
def create_guest(body: GuestIn, request: Request, x_mmv_key: str = Header(default="")):
    if not _is_admin(x_mmv_key, request):
        raise HTTPException(403, "Clé admin incorrecte")
    if body.plan not in LIMITS:
        raise HTTPException(400, "Offre inconnue")
    days = max(1, min(int(body.days), 90))
    label = "".join(c for c in body.label if c.isalnum() or c in " -_.@")[:40].strip() or "invité"
    gid = secrets.token_hex(3)
    exp = int(_time.time()) + days * 86400
    payload = "|".join([body.plan, str(exp), gid, label])
    code = "inv_" + _b64(payload.encode()) + "." + _sign(payload)
    return {"code": code, "id": gid, "plan": body.plan, "expires": exp, "label": label, "link": SITE_URL + "/compte.html?invite=" + urllib.parse.quote(code)}


def _entitlement_from_guest(code):
    try:
        raw, sig = code[4:].split(".", 1)
        payload = _unb64(raw).decode()
        plan, exp, gid, label = payload.split("|", 3)
        exp = int(exp)
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Code invalide")
    if not hmac.compare_digest(sig.encode(), _sign(payload).encode()) or plan not in LIMITS:
        raise HTTPException(404, "Code inconnu")
    revoked = [x.strip() for x in os.environ.get("MMV_REVOKED_GUESTS", "").split(",") if x.strip()]
    active = exp > _time.time() and gid not in revoked
    return {"active": active, "plan": plan, "limits": LIMITS[plan], "guest": True, "expires": exp, "label": label, "recurring": False}


@app.get("/api/entitlement")
def entitlement(request: Request, license: str = "", x_mmv_key: str = Header(default="")):
    if _is_admin(x_mmv_key, request):
        return {"active": True, "plan": "admin", "admin": True, "limits": {"mixes": None, "maxTracks": 50, "beatmaker": True}}
    license = license.strip()
    if not license:
        return {"active": False, "plan": None}
    if license.startswith("inv_"):
        if len(license) > 400 or not os.environ.get("MMV_ADMIN_KEY"):
            raise HTTPException(400, "Code invalide")
        return _entitlement_from_guest(license)
    return _entitlement_from_session(license)
