"""MixMyVibz — comptes par lien magique (sans mot de passe, sans base de données).

Principe
- L'utilisateur donne son email : on lui envoie un lien valable 15 minutes. Le premier lien crée le compte, les suivants le retrouvent (inscription et connexion = même geste).
- Le lien donne une session signée de 30 jours (jeton stocké dans le navigateur). Rien n'est stocké sur le serveur : tout est signé avec MMV_AUTH_SECRET (à défaut MMV_ADMIN_KEY).
- L'offre et les paiements se retrouvent chez Stripe à partir de l'email : abonnement actif -> offre ; paiement « Un titre » -> offre ; historique des paiements.

Variables Render
- RESEND_API_KEY  : clé d'envoi d'emails (resend.com). Sans elle, la demande de lien renvoie 503.
- MMV_MAIL_FROM   : expéditeur, par défaut « MixMyVibz <login@mixmyvibz.com> » (le domaine doit être vérifié chez Resend).
- MMV_AUTH_SECRET : secret de signature (sinon MMV_ADMIN_KEY).
- MMV_MAIL_DEBUG=1 : en test seulement, renvoie le lien dans la réponse au lieu d'envoyer un email.
"""
import base64, hashlib, hmac, json, os, re, secrets, time, urllib.error, urllib.parse, urllib.request

from fastapi import Header, HTTPException, Request
from pydantic import BaseModel

SITE_URL = os.environ.get("MMV_SITE_URL", "https://mixmyvibz.com").rstrip("/")
LINK_TTL = 15 * 60
SESSION_TTL = 30 * 86400
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,}$")
_HITS = {}  # limite de demandes (mémoire du processus)
_USED = {}  # liens déjà utilisés (au mieux : se vide au redémarrage, mais le lien expire en 15 min)

# Reprend les identifiants de prix et limites du serveur (mêmes valeurs que main.py).
PRICES = {
    "single": os.environ.get("STRIPE_PRICE_SINGLE", "price_1URfcERNHlUwN64AR0tlmldZ"),
    "pro": os.environ.get("STRIPE_PRICE_PRO", "price_1URfcERNHlUwN64A65XA1UfA"),
    "studio": os.environ.get("STRIPE_PRICE_STUDIO", "price_1URfcERNHlUwN64AdrtJfABT"),
}
LIMITS = {
    "single": {"mixes": 1, "maxTracks": 8, "beatmaker": False},
    "pro": {"mixes": 15, "maxTracks": 20, "beatmaker": True},
    "studio": {"mixes": None, "maxTracks": 50, "beatmaker": True},
}
RANK = {"single": 1, "pro": 2, "studio": 3}


def _secret():
    s = os.environ.get("MMV_AUTH_SECRET") or os.environ.get("MMV_ADMIN_KEY") or ""
    if not s:
        raise HTTPException(503, "Comptes non configurés sur ce serveur")
    return s.encode()


def _b64(b):
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload):
    return _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest()[:20])


def make_token(kind, email, ttl):
    payload = "|".join([kind, email, str(int(time.time()) + ttl), secrets.token_hex(4)])
    return kind + "_" + _b64(payload.encode()) + "." + _sign(payload)


def read_token(kind, token):
    """Renvoie (email, nonce) si le jeton est valide, sinon lève une erreur 401."""
    bad = HTTPException(401, "Lien invalide ou expiré. Demande-en un nouveau.")
    try:
        if not token.startswith(kind + "_") or len(token) > 600:
            raise bad
        raw, sig = token[len(kind) + 1:].split(".", 1)
        payload = _unb64(raw).decode()
        k, email, exp, nonce = payload.split("|", 3)
        exp = int(exp)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        raise bad
    if k != kind or not hmac.compare_digest(sig.encode(), _sign(payload).encode()) or exp < time.time():
        raise bad
    return email, nonce


# ---------- Stripe (lecture seule, à partir de l'email) ----------
def _stripe_get(path, params=None):
    key = os.environ.get("STRIPE_SECRET_KEY")
    if not key:
        return None
    url = "https://api.stripe.com" + path + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise HTTPException(502, "Stripe : " + (e.read().decode()[:160] if e.fp else str(e.code)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, "Stripe injoignable : " + str(e)[:100])


def _plan_of_price(price_id):
    for p, pid in PRICES.items():
        if pid == price_id:
            return p
    return None


def billing_for(email):
    """Offre active + historique des paiements d'un email, lus chez Stripe."""
    best, payments, renew = None, [], None
    custs = _stripe_get("/v1/customers", {"email": email, "limit": 10})
    if custs is None:
        return {"active": False, "plan": None, "payments": [], "stripe": False}
    for c in custs.get("data", []):
        subs = _stripe_get("/v1/subscriptions", {"customer": c["id"], "status": "all", "limit": 10, "expand[]": "data.items.data.price"}) or {}
        for s in subs.get("data", []):
            if s.get("status") not in ("active", "trialing"):
                continue
            for it in (s.get("items") or {}).get("data", []):
                p = _plan_of_price((it.get("price") or {}).get("id"))
                if p and (best is None or RANK[p] > RANK[best]):
                    best, renew = p, s.get("current_period_end") or (it or {}).get("current_period_end")
    sess = _stripe_get("/v1/checkout/sessions", {"customer_details[email]": email, "limit": 20}) or {}
    for s in sess.get("data", []):
        if s.get("payment_status") != "paid":
            continue
        payments.append({"date": s.get("created"), "amount": (s.get("amount_total") or 0) / 100, "currency": (s.get("currency") or "").upper(), "mode": s.get("mode")})
        if s.get("mode") == "payment" and best is None:
            best = "single"  # paiement unique « Un titre » : valable tant qu'il n'est pas consommé (le décompte des mix reste côté navigateur pour l'instant)
    return {"active": best is not None, "plan": best, "limits": LIMITS.get(best), "recurring": best in ("pro", "studio"), "renews": renew, "payments": payments, "stripe": True}


# ---------- envoi d'email ----------
def send_login_email(email, link):
    key = os.environ.get("RESEND_API_KEY")
    if os.environ.get("MMV_MAIL_DEBUG") == "1":
        return {"debug_link": link}
    if not key:
        raise HTTPException(503, "L'envoi d'emails n'est pas encore configuré sur le serveur.")
    html = (
        "<div style=\"font:16px/1.5 system-ui,sans-serif;color:#111\"><h2 style=\"margin:0 0 12px\">Ta connexion à MixMyVibz</h2>"
        "<p>Clique sur le bouton pour te connecter. Le lien est valable 15 minutes et ne marche qu'une fois.</p>"
        f"<p><a href=\"{link}\" style=\"display:inline-block;background:#6c5ce7;color:#fff;padding:12px 20px;border-radius:10px;text-decoration:none;font-weight:600\">Me connecter</a></p>"
        "<p style=\"color:#666;font-size:13px\">Si tu n'as rien demandé, ignore ce message : personne ne peut se connecter sans ce lien.</p></div>"
    )
    body = json.dumps({"from": os.environ.get("MMV_MAIL_FROM", "MixMyVibz <login@mixmyvibz.com>"), "to": [email], "subject": "Ton lien de connexion MixMyVibz", "html": html, "text": "Connexion à MixMyVibz (valable 15 minutes) : " + link}).encode()
    req = urllib.request.Request("https://api.resend.com/emails", data=body, headers={"Authorization": "Bearer " + key, "content-type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=15).read()
    except urllib.error.HTTPError as e:
        raise HTTPException(502, "Envoi de l'email impossible (" + str(e.code) + "). Vérifie le domaine d'envoi chez Resend.")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, "Envoi de l'email impossible : " + str(e)[:100])
    return {}


class EmailIn(BaseModel):
    email: str


class TokenIn(BaseModel):
    token: str


def _limit(key, n, per):
    now = time.time()
    hits = [t for t in _HITS.get(key, []) if now - t < per]
    if len(hits) >= n:
        raise HTTPException(429, "Trop de demandes. Réessaie dans quelques minutes.")
    hits.append(now)
    _HITS[key] = hits
    if len(_HITS) > 5000:
        _HITS.clear()


def _bearer(authorization):
    a = (authorization or "").strip()
    return a[7:] if a.lower().startswith("bearer ") else ""


def mount(app):
    @app.post("/api/auth/request")
    def auth_request(body: EmailIn, request: Request):
        email = body.email.strip().lower()
        if not EMAIL_RE.match(email):
            raise HTTPException(400, "Adresse email invalide")
        ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "")).split(",")[0].strip()
        _limit("e:" + email, 4, 3600)
        _limit("i:" + ip, 12, 3600)
        link = SITE_URL + "/compte.html?login=" + urllib.parse.quote(make_token("lnk", email, LINK_TTL))
        extra = send_login_email(email, link)
        return {"ok": True, **extra}

    @app.post("/api/auth/verify")
    def auth_verify(body: TokenIn):
        email, nonce = read_token("lnk", body.token.strip())
        if nonce in _USED:
            raise HTTPException(401, "Ce lien a déjà servi. Demande-en un nouveau.")
        _USED[nonce] = time.time()
        if len(_USED) > 5000:
            _USED.clear()
        return {"session": make_token("ses", email, SESSION_TTL), "email": email}

    @app.get("/api/auth/me")
    def auth_me(authorization: str = Header(default="")):
        email, _ = read_token("ses", _bearer(authorization))
        return {"email": email, **billing_for(email)}
