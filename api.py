from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from typing import Optional
import openai
import os
import json
import psycopg2
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from prometheus_fastapi_instrumentator import Instrumentator
from dateutil import parser
from datetime import datetime
import re
from datetime import timedelta
import requests
from db import get_db_connection

load_dotenv()
client = OpenAI()

# Initialisation FastAPI
app = FastAPI()
 
# Monitoring via Prometheus
try:
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
except ImportError:
    print("prometheus_fastapi_instrumentator is not installed. Monitoring not enabled.")

# CORS pour Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration OpenAI
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

# Connexion PostgreSQL
conn = None
cursor = None

def ensure_db_connection():
    global conn, cursor
    try:
        if conn is None or conn.closed:
            conn = get_db_connection()
        if cursor is None or cursor.closed:
            cursor = conn.cursor() if conn else None
        if conn is None or cursor is None:
            raise HTTPException(status_code=503, detail="Base de données indisponible")
    except HTTPException:
        raise
    except Exception as e:
        print("Erreur reconnexion base :", e)
        conn = None
        cursor = None
        raise HTTPException(status_code=503, detail="Base de données indisponible")

# Pydantic models
class AskRequest(BaseModel):
    message: str
    user_id: str

class Reservation(BaseModel):
    date: str
    hour: str
    reserved_by: Optional[str] = ""

class Absence(BaseModel):
    name: str
    date: str
    
class Message(BaseModel):
    message: str
    user_id: str
    history: List[Dict[str, str]] = []
    
@app.get("/")
def root():
    return {"message": "API bot IA en ligne 👋"}

def get_or_create_thread(user_id: str) -> str:
    ensure_db_connection()
    if cursor is None:
        raise HTTPException(status_code=503, detail="Base de données indisponible")
    cursor.execute("SELECT thread_id FROM user_threads WHERE user_id = %s", (user_id,))
    result = cursor.fetchone()
    if result:
        return result[0], False
    else:
        thread = client.beta.threads.create(metadata={"user_id": user_id})
        thread_id = thread.id
        user_instructions = get_user_instructions(user_id)
        # Les instructions spécifiques seront injectées plus tard dans run.create
        pass
        cursor.execute("INSERT INTO user_threads (user_id, thread_id) VALUES (%s, %s)", (user_id, thread_id))
        conn.commit()
        return thread_id, True

def cancel_active_runs(thread_id: str):
    try:
        existing_runs = client.beta.threads.runs.list(thread_id=thread_id).data
        for run in existing_runs:
            if run.status in ["queued", "in_progress", "requires_action"]:
                print(f"Annulation du run actif : {run.id}")
                client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
    except Exception as e:
        print(f"Erreur lors de l’annulation des runs actifs : {e}")

def preprocess_date(text_date: str) -> str:
    text_date = text_date.lower().strip()

    today = datetime.now()
    if "après-demain" in text_date:
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")
    elif "demain" in text_date:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "aujourd" in text_date:
        return today.strftime("%Y-%m-%d")

    return text_date

def normalize_date(text_date: str) -> str:
    now = datetime.now()
    default = datetime(now.year, now.month, now.day)
    clean_date = preprocess_date(text_date)
    parsed = parser.parse(clean_date, default=default)
    if parsed.year != now.year and str(now.year) not in text_date:
        parsed = parsed.replace(year=now.year)
    return parsed.date().isoformat()

# Endpoint principal : question posée à l'agent
@app.post("/ask_agent")
async def ask_agent(req: Message):
    thread_id, is_new_thread = get_or_create_thread(req.user_id)
    cancel_active_runs(thread_id)

    if is_new_thread:
        for hist_msg in req.history:
            client.beta.threads.messages.create(
                thread_id=thread_id,
                role=hist_msg.get("role", "user"),
                content=hist_msg["content"],
                metadata={"author": hist_msg["author"]}
            )

    last_user_author = None
    for _m in reversed(req.history):
        if _m.get("role") == "user" and _m.get("author"):
            last_user_author = _m.get("author")
            break
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=req.message,
        metadata={
            "author": req.user_id,
            "author_name": last_user_author or req.user_id
        }
    )

    user_instructions = get_user_instructions(req.user_id)
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
        instructions=user_instructions,
        metadata={"user_id": req.user_id}
    )

    while run.status != "completed":
        if run.status == "requires_action":
            tools_calls = run.required_action.submit_tool_outputs.tool_calls
            outputs = []
            result = None
            for call in tools_calls:
                func_name = call.function.name
                arguments = json.loads(call.function.arguments)
                if func_name == "book_box":
                    reserved_by = arguments.get("reserved_by") or next((msg["author"] for msg in reversed(req.history) if msg["role"] == "user"), req.user_id)
                    result = book_box_logic(date=arguments["date"], hour=arguments["hour"], reserved_by=reserved_by)
                elif func_name == "report_absence":
                    try:
                        parsed_date = normalize_date(arguments["date"])
                        if not arguments.get("name"):
                            _last_u = next((m.get("author") for m in reversed(req.history) if m.get("role") == "user" and m.get("author")), None)
                            name = _last_u or req.user_id
                        else:
                            name = arguments.get("name")
                        result = report_absence_logic(name=name, date=parsed_date)
                    except Exception as e:
                        result = f"Erreur dans la compréhension de la date : {e}"
                elif func_name == "generate_image":
                    try:
                        image = client.images.generate(
                            model="dall-e-3",
                            prompt=arguments["prompt"],
                            size="1024x1024",
                            n=1
                        )
                        result = image.data[0].url
                    except Exception as e:
                        result = f"Erreur DALL·E : {e}"
                elif func_name == "list_absences":
                    absences = get_absences()
                    if absences and isinstance(absences[0], dict) and "message" in absences[0]:
                        result = absences[0]["message"]
                    else:
                        result = "\n".join(f"{a['name']} - {a['date']}" for a in absences)
                elif func_name == "list_reservations":
                    reservations = get_reservations()
                    if reservations and isinstance(reservations[0], dict) and "message" in reservations[0]:
                        result = reservations[0]["message"]
                    else:
                        result = "\n".join(f"{r['date']} à {r['hour']} réservé par {r['reserved_by']}" for r in reservations)
                elif func_name == "update_reservation":
                    result = update_reservation(
                        res_id=int(arguments["res_id"]),
                        date=arguments.get("date"),
                        hour=arguments.get("hour"),
                        reserved_by=arguments.get("reserved_by")
                    )["message"]
                elif func_name == "update_absence":
                    result = update_absence(
                        abs_id=int(arguments["abs_id"]),
                        name=arguments.get("name"),
                        date=arguments.get("date")
                    )["message"]
                elif func_name == "search_brave":
                    result = search_brave(arguments["query"])

                outputs.append({"tool_call_id": call.id, "output": result})
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=outputs
            )
        else:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    return {
        "response": messages.data[0].content[0].text.value,
        "thread_id": thread_id,
        "turn_id": messages.data[0].id
    }

# Logic métier pour réserver un box
def book_box_logic(date: str, hour: str, reserved_by: str = None):
    ensure_db_connection()
    if not reserved_by:
        reserved_by = "Inconnu"
    cursor.execute("""
        INSERT INTO reservations (date, hour, reserved_by)
        VALUES (%s, %s, %s)
    """, (date, hour, reserved_by))
    conn.commit()
    return "done"

# Logic métier pour déclarer une absence
def report_absence_logic(name: str, date: str):
    ensure_db_connection()
    cursor.execute("""
        INSERT INTO absences (name, date)
        VALUES (%s, %s)
    """, (name, date))
    conn.commit()
    return "done"

# Endpoint pour voir toutes les réservations (Streamlit)
@app.get("/reservations")
def get_reservations():
    ensure_db_connection()
    if cursor is None:
        raise HTTPException(status_code=500, detail="Base de données non accessible")
    cursor.execute("SELECT date, hour, reserved_by FROM reservations ORDER BY date, hour")
    results = cursor.fetchall()
    if not results:
        return [{"message": "Aucune réservation à venir."}]
    response = []
    for r in results:
        try:
            date = r[0].isoformat() if r[0] else "Date inconnue"
            hour = r[1] or "Heure inconnue"
            reserved_by = r[2] or "Inconnu"
            response.append({"date": date, "hour": hour, "reserved_by": reserved_by})
        except Exception as e:
            print("Erreur lors du traitement d'une ligne de réservation :", e)
    return response

# Endpoint pour ajouter une réservation manuellement (Streamlit)
@app.post("/book_box")
def book_box_manual(data: Reservation):
    return book_box_logic(data.date, data.hour, data.reserved_by)

# Endpoint pour enregistrer une absence manuellement
@app.post("/report_absence")
def report_absence_manual(data: Absence):
    return report_absence_logic(data.name, data.date)

@app.get("/list_absences")
def get_absences():
    ensure_db_connection()
    try:
        cursor.execute("SELECT name, date FROM absences WHERE date >= CURRENT_DATE ORDER BY date, name")
        results = cursor.fetchall()
        if not results:
            return [{"message": "Aucune absence à venir enregistrée."}]
        return [{"name": r[0], "date": r[1].isoformat()} for r in results]
    except Exception as e:
        if conn:
            conn.rollback() 
        print("Erreur dans get_absences :", e)
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des absences") from e

@app.get("/list_reservations")
def list_reservations():
    ensure_db_connection()
    try:
        cursor.execute("SELECT date, hour, reserved_by FROM reservations WHERE (date > CURRENT_DATE OR (date = CURRENT_DATE AND hour >= TO_CHAR(NOW(), 'HH24:MI'))) ORDER BY date, hour")
        results = cursor.fetchall()
        if not results:
            return [{"message": "Aucune réservation à venir."}]
        return [{"date": r[0].isoformat() if r[0] else None, "hour": r[1], "reserved_by": r[2] or "Inconnu"} for r in results]
    except Exception as e:
        if conn:
            conn.rollback()
        print("Erreur dans list_reservations :", e)
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des réservations") from e

@app.put("/update_reservation")
def update_reservation(res_id: int, date: Optional[str] = None, hour: Optional[str] = None, reserved_by: Optional[str] = None):
    ensure_db_connection()
    updates = []
    values = []

    if date:
        updates.append("date = %s")
        values.append(date)
    if hour:
        updates.append("hour = %s")
        values.append(hour)
    if reserved_by:
        updates.append("reserved_by = %s")
        values.append(reserved_by)

    if not updates:
        raise HTTPException(status_code=400, detail="Aucune mise à jour spécifiée.")

    values.append(res_id)
    query = f"UPDATE reservations SET {', '.join(updates)} WHERE id = %s"
    cursor.execute(query, values)
    conn.commit()
    return {"message": "Réservation mise à jour avec succès"}

@app.put("/update_absence")
def update_absence(abs_id: int, name: Optional[str] = None, date: Optional[str] = None):
    ensure_db_connection()
    updates = []
    values = []

    if name:
        updates.append("name = %s")
        values.append(name)
    if date:
        updates.append("date = %s")
        values.append(date)

    if not updates:
        raise HTTPException(status_code=400, detail="Aucune mise à jour spécifiée.")

    values.append(abs_id)
    query = f"UPDATE absences SET {', '.join(updates)} WHERE id = %s"
    cursor.execute(query, values)
    conn.commit()
    return {"message": "Absence mise à jour avec succès"}

def search_brave(query: str) -> str:
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": os.getenv("BRAVE_API_KEY")
    }
    params = {"q": query, "count": 5}

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if "web" in data and "results" in data["web"]:
        results = data["web"]["results"]
        formatted = "\n".join([f"{r['title']} - {r['url']}" for r in results])
        return formatted
    else:
        return "Aucun résultat trouvé via Brave Search."

def get_user_instructions(user_id: str) -> Optional[str]:
    ensure_db_connection()
    try:
        cursor.execute("SELECT instructions FROM user_profiles WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        if result:
            return result[0]
    except Exception as e:
        print(f"Erreur lors de la récupération des instructions utilisateur : {e}")
    return None
