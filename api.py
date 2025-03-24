from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
import os
import json
import psycopg2
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

# Initialisation FastAPI
app = FastAPI()

# CORS pour Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

# Connexion PostgreSQL
try:
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT")
    )
    cursor = conn.cursor()
except Exception as e:
    print("Erreur connexion PostgreSQL :", e)
    conn = None
    cursor = None

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

# Endpoint principal : question posée à l'agent
@app.post("/ask_agent")
async def ask_agent(req: AskRequest):
    thread = openai.beta.threads.create()
    thread_id = thread.id

    openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=req.message
    )
    run = openai.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
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
                    result = book_box_logic(**arguments)
                elif func_name == "report_absence":
                    result = report_absence_logic(**arguments)
                outputs.append({"tool_call_id": call.id, "output": result})
            run = openai.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=outputs
            )
        else:
            run = openai.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )

    messages = openai.beta.threads.messages.list(thread_id=thread_id)
    return {"response": messages.data[0].content[0].text.value}

# Logic métier pour réserver un box
def book_box_logic(date: str, hour: str, reserved_by: str = "Agent"):
    cursor.execute("""
        INSERT INTO reservations (date, hour, reserved_by)
        VALUES (%s, %s, %s)
    """, (date, hour, reserved_by))
    conn.commit()
    return f"Box réservé le {date} à {hour} par {reserved_by}."

# Logic métier pour déclarer une absence
def report_absence_logic(name: str, date: str):
    cursor.execute("""
        INSERT INTO absences (name, date)
        VALUES (%s, %s)
    """, (name, date))
    conn.commit()
    return f"Absence enregistrée pour {name} le {date}."

# Endpoint pour voir toutes les réservations (Streamlit)
@app.get("/reservations")
def get_reservations():
    if cursor is None:
        raise HTTPException(status_code=500, detail="Base de données non accessible")
    cursor.execute("SELECT date, hour, reserved_by FROM reservations ORDER BY date, hour")
    results = cursor.fetchall()
    return [
        {"date": r[0].isoformat(), "hour": r[1], "reserved_by": r[2]} for r in results
    ]

# Endpoint pour ajouter une réservation manuellement (Streamlit)
@app.post("/book_box")
def book_box_manual(data: Reservation):
    return book_box_logic(data.date, data.hour, data.reserved_by)

# Endpoint pour enregistrer une absence manuellement
@app.post("/report_absence")
def report_absence_manual(data: Absence):
    return report_absence_logic(data.name, data.date)

@app.get("/absences")
def get_absences():
    try:
        cursor.execute("SELECT name, date FROM absences ORDER BY date, name")
        results = cursor.fetchall()
        return [{"name": r[0], "date": r[1].isoformat()} for r in results]
    except Exception as e:
        conn.rollback() 
        print("Erreur dans get_absences :", e)
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des absences") from e
