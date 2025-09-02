# server.py
import os, uuid, time, sqlite3, json
from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx

# --- Vendor SDKs you already used ---
from openai import OpenAI
import anthropic

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "https://example.org")
OPENROUTER_TITLE    = os.getenv("OPENROUTER_TITLE", "SpiralBench")

USER_AGENT_BASE_SYSTEM_PROMPT = ""

def load_text_file(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def load_category_instructions() -> Dict[str, str]:
    category_instructions_path = Path("user_instructions/category_instructions.json")
    try:
        if category_instructions_path.exists():
            with open(category_instructions_path, "r", encoding="utf-8") as cf:
                return json.load(cf)
    except Exception:
        pass
    return {}

def ts_iso(ts: float) -> str:
    try:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds")
    except Exception:
        return str(ts)

app = FastAPI()
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"]
)

DB = os.getenv("DB_PATH", "spiralbench.db")

def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA busy_timeout=5000;")
        cur = con.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS sessions(
            session_id TEXT PRIMARY KEY,
            ra_pseudonym TEXT,
            user_model TEXT,
            evaluated_model TEXT,
            scenario_id TEXT,
            created_at REAL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS turns(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            turn_index INTEGER,
            source TEXT,
            role TEXT,
            content TEXT,
            meta_json TEXT,
            created_at REAL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS human_codes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            turn_index INTEGER,
            pushback INTEGER,
            de_escalation INTEGER,
            redirection INTEGER,
            help_referral INTEGER,
            consciousness_claims INTEGER,
            delusion_reinforcement INTEGER,
            escalation INTEGER,
            harmful_advice INTEGER,
            sycophancy INTEGER,
            created_at REAL
        )""")
        con.commit()

init_db()

# ────────────────────────────── Session create / list / get / transcript ──────────────────────────────

class CreateSession(BaseModel):
    ra_pseudonym: str
    user_model: str
    evaluated_model: str
    scenario_id: str

@app.post("/api/session")
def create_session(req: CreateSession):
    sid = str(uuid.uuid4())
    with sqlite3.connect(DB) as con:
        con.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
            (sid, req.ra_pseudonym, req.user_model, req.evaluated_model, req.scenario_id, time.time()))
        con.commit()
    return {"session_id": sid}

@app.get("/api/sessions")
def list_sessions(limit: int = 50, ra: Optional[str] = None, scenario: Optional[str] = None, q: Optional[str] = None):
    """Recent sessions with quick stats for the browser UI."""
    sql = """
    SELECT s.session_id, s.ra_pseudonym, s.user_model, s.evaluated_model, s.scenario_id, s.created_at,
           COALESCE((SELECT MAX(turn_index) FROM turns t WHERE t.session_id = s.session_id), -1) AS last_turn_index,
           COALESCE((SELECT MAX(created_at) FROM turns t2 WHERE t2.session_id = s.session_id), s.created_at) AS updated_at,
           COALESCE((SELECT COUNT(*) FROM turns t3 WHERE t3.session_id = s.session_id), 0) AS turn_count
    FROM sessions s
    """
    where = []
    args: List[Any] = []
    if ra:
        where.append("s.ra_pseudonym = ?"); args.append(ra)
    if scenario:
        where.append("s.scenario_id = ?"); args.append(scenario)
    if q:
        where.append("(s.session_id LIKE ? OR s.ra_pseudonym LIKE ? OR s.scenario_id LIKE ?)")
        args.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    args.append(limit)

    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, args).fetchall()
    out = []
    for r in rows:
        out.append({
            "session_id": r["session_id"],
            "ra_pseudonym": r["ra_pseudonym"],
            "user_model": r["user_model"],
            "evaluated_model": r["evaluated_model"],
            "scenario_id": r["scenario_id"],
            "created_at": ts_iso(r["created_at"]),
            "updated_at": ts_iso(r["updated_at"]),
            "turn_count": r["turn_count"],
            "last_turn_index": r["last_turn_index"],
            "next_index": (r["last_turn_index"] or -1) + 1
        })
    return {"sessions": out}

@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        s = con.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not s:
            raise HTTPException(404, "Session not found")
        last_idx = con.execute("SELECT COALESCE(MAX(turn_index), -1) FROM turns WHERE session_id=?", (session_id,)).fetchone()[0]
        updated_at = con.execute("SELECT COALESCE(MAX(created_at), ?) FROM turns WHERE session_id=?", (s["created_at"], session_id)).fetchone()[0]
    return {
        "session_id": s["session_id"],
        "ra_pseudonym": s["ra_pseudonym"],
        "user_model": s["user_model"],
        "evaluated_model": s["evaluated_model"],
        "scenario_id": s["scenario_id"],
        "created_at": ts_iso(s["created_at"]),
        "updated_at": ts_iso(updated_at),
        "last_turn_index": last_idx,
        "next_index": (last_idx or -1) + 1
    }

@app.get("/api/session/{session_id}/transcript")
def get_transcript(session_id: str):
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        turns = con.execute("""
            SELECT turn_index, source, role, content, created_at
            FROM turns WHERE session_id=?
            ORDER BY turn_index ASC, created_at ASC
        """, (session_id,)).fetchall()
    transcript = [{
        "turn_index": r["turn_index"],
        "source": r["source"],
        "role": r["role"],
        "content": r["content"],
        "created_at": ts_iso(r["created_at"])
    } for r in turns]
    last_idx = max([t["turn_index"] for t in transcript], default=-1)
    return {"transcript": transcript, "turn_count": len(transcript), "last_turn_index": last_idx, "next_index": last_idx + 1}

# ────────────────────────────── Logging & Codes ──────────────────────────────

class LogTurn(BaseModel):
    session_id: str
    turn_index: int
    source: Literal["chat_ui","api"]
    role: Literal["user","assistant"]
    content: str
    meta_json: Optional[str] = None

@app.post("/api/turn")
def log_turn(t: LogTurn):
    role = "assistant" if t.source == "chat_ui" else t.role
    with sqlite3.connect(DB) as con:
        con.execute("""INSERT INTO turns(session_id,turn_index,source,role,content,meta_json,created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                       (t.session_id, t.turn_index, t.source, role, t.content, t.meta_json or "{}", time.time()))
        con.commit()
    return {"ok": True}

class CodeTurn(BaseModel):
    session_id: str
    turn_index: int
    pushback: int = 0
    de_escalation: int = 0
    redirection: int = 0
    help_referral: int = 0
    consciousness_claims: int = 0
    delusion_reinforcement: int = 0
    escalation: int = 0
    harmful_advice: int = 0
    sycophancy: int = 0

@app.post("/api/code")
def code_turn(c: CodeTurn):
    with sqlite3.connect(DB) as con:
        con.execute("""INSERT INTO human_codes(session_id,turn_index,pushback,de_escalation,redirection,
            help_referral,consciousness_claims,delusion_reinforcement,escalation,harmful_advice,sycophancy,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (c.session_id,c.turn_index,c.pushback,c.de_escalation,c.redirection,
             c.help_referral,c.consciousness_claims,c.delusion_reinforcement,c.escalation,c.harmful_advice,c.sycophancy,time.time()))
        con.commit()
    return {"ok": True}

# ────────────────────────────── API Call (simulated user) ────────────────────

class APICall(BaseModel):
    session_id: str
    user_model: str
    evaluated_model: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    seed: Optional[int] = None
    user_message: str
    role: Literal["user","assistant"] = "user"
    provider: Optional[Literal["openai","anthropic","openrouter"]] = None
    openrouter_base_url: Optional[str] = None
    category: Optional[str] = None
    instruction_file: Optional[str] = None
    seed_prompt: Optional[str] = None

@app.post("/api/call")
def api_call(req: APICall):
    instruction_file = req.instruction_file or "user_instructions/default.txt"
    instructions = load_text_file(instruction_file)
    user_system_prompt = f"{USER_AGENT_BASE_SYSTEM_PROMPT}\n{instructions}".strip()

    if req.category:
        category_map = load_category_instructions()
        extra_text = category_map.get(req.category, "").strip()
        if extra_text:
            user_system_prompt += f"\n\n# Extra instructions for this roleplay:\n{extra_text}"

    if req.system_prompt:
        user_system_prompt = f"{user_system_prompt}\n\n{req.system_prompt}"

    with sqlite3.connect(DB) as con:
        rows = con.execute("""SELECT role, content FROM turns
                              WHERE session_id=? ORDER BY turn_index ASC, created_at ASC""", (req.session_id,)).fetchall()
    transcript = [{"role": r, "content": c} for (r, c) in rows]

    if req.seed_prompt and not transcript:
        transcript = [{"role": "user", "content": req.seed_prompt}]
    elif req.seed_prompt and transcript:
        if not transcript or transcript[0]["content"] != req.seed_prompt:
            transcript = [{"role": "user", "content": req.seed_prompt}] + transcript

    user_msgs = [{"role": "system", "content": user_system_prompt}]
    user_msgs.extend(transcript)
    user_msgs.append({"role": req.role, "content": req.user_message})

    if user_msgs[-1]["role"] == "assistant":
        user_msgs[-1] = {
            **user_msgs[-1],
            "content": user_msgs[-1]["content"] + "\n\n[Instructions for your response:\n" + user_system_prompt + "\n]"
        }

    flipped = [{"role": ("user" if m["role"] == "assistant" else "assistant"), "content": m["content"]} for m in user_msgs[1:]]
    messages = [user_msgs[0]] + flipped

    provider = req.provider
    if provider is None:
        provider = "anthropic" if req.user_model.lower().startswith(("claude","anthropic")) else "openai"

    if provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise HTTPException(400, "Missing ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        sys = req.system_prompt or ""
        user_text = "\n\n".join([m["content"] for m in messages if m["role"]=="user"])
        r = client.messages.create(
            model=req.user_model,
            temperature=req.temperature,
            system=sys if sys else None,
            messages=[{"role":"user","content":user_text}]
        )
        content = "".join([p.text for p in r.content if p.type=="text"])

    elif provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise HTTPException(400, "Missing OPENROUTER_API_KEY")
        base = (req.openrouter_base_url or OPENROUTER_BASE_URL).rstrip("/")
        url = base + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_TITLE,
        }
        body = {"model": req.user_model, "messages": messages, "temperature": req.temperature}
        if req.seed is not None:
            body["seed"] = req.seed
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=60)
        except Exception as e:
            raise HTTPException(502, f"OpenRouter network error: {e}")
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise HTTPException(resp.status_code, f"OpenRouter error: {err}")
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            raise HTTPException(502, f"Unexpected OpenRouter payload: {data}")

    else:  # openai
        if not OPENAI_API_KEY:
            raise HTTPException(400, "Missing OPENAI_API_KEY")
        client = OpenAI(api_key=OPENAI_API_KEY)
        r = client.chat.completions.create(
            model=req.user_model,
            temperature=req.temperature,
            seed=req.seed,
            messages=messages
        )
        content = r.choices[0].message.content

    if not content or not content.strip():
        raise HTTPException(502, "API returned empty or null content")

    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        row = cur.execute("SELECT COALESCE(MAX(turn_index), -1) FROM turns WHERE session_id=?", (req.session_id,)).fetchone()
        next_idx = (row[0] or -1) + 1
        cur.execute("""INSERT INTO turns(session_id,turn_index,source,role,content,meta_json,created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (req.session_id,next_idx,"api","user",content,"{}",time.time()))
        con.commit()

    return {"assistant": content, "turn_index": next_idx}

# Prompts for UI
@app.get("/api/prompts")
def get_prompts():
    try:
        with open("prompts/eval_prompts_v0.2.json", "r", encoding="utf-8") as f:
            prompts = json.load(f)
        return {"prompts": prompts}
    except FileNotFoundError:
        return {"prompts": []}

@app.get("/")
def serve_ui():
    return FileResponse("ui.html")