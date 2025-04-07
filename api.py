import os
import json
import logging
import asyncio
from datetime import date
from typing import List, Dict, Optional
import databases
import sqlalchemy
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI 
from prometheus_fastapi_instrumentator import Instrumentator
from dateutil import parser as date_parser 

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
if not OPENAI_API_KEY or not ASSISTANT_ID:
    logger.error("Variables d'environnement OpenAI manquantes (OPENAI_API_KEY, OPENAI_ASSISTANT_ID)")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

DATABASE_URL = (
    f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

user_threads = sqlalchemy.Table(
    "user_threads",
    metadata,
    sqlalchemy.Column("user_id", sqlalchemy.String(255), primary_key=True),
    sqlalchemy.Column("thread_id", sqlalchemy.String(255), nullable=False, unique=True),
)

reservations = sqlalchemy.Table(
    "reservations",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("date", sqlalchemy.Date, nullable=False),
    sqlalchemy.Column("hour", sqlalchemy.String(5), nullable=False),
    sqlalchemy.Column("reserved_by", sqlalchemy.String(255), server_default='Agent'),
    sqlalchemy.Index("idx_reservations_date_hour", "date", "hour"),
)

absences = sqlalchemy.Table(
    "absences",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(255), nullable=False),
    sqlalchemy.Column("date", sqlalchemy.Date, nullable=False),
    sqlalchemy.Index("idx_absences_date_name", "date", "name"),
)


app = FastAPI(title="API Bot IA", version="1.0")

try:
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
except Exception as e:
    logger.warning(f"prometheus_fastapi_instrumentator non disponible. Monitoring désactivé. Erreur: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    try:
        await database.connect()
        logger.info("Connexion à la base de données établie.")
    except Exception as e:
        logger.critical(f"Échec de la connexion à la base de données au démarrage: {e}")

@app.on_event("shutdown")
async def shutdown():
    try:
        await database.disconnect()
        logger.info("Connexion à la base de données fermée.")
    except Exception as e:
        logger.error(f"Erreur lors de la déconnexion de la base de données: {e}")

# --- Pydantic Models ---
class Message(BaseModel):
    message: str
    user_id: str

class ReservationInput(BaseModel):
    date: str
    hour: str
    reserved_by: Optional[str] = "Interface Manuelle"

class ReservationOutput(BaseModel):
    date: date
    hour: str
    reserved_by: Optional[str]

class AbsenceInput(BaseModel):
    name: str
    date: str

class AbsenceOutput(BaseModel):
    name: str
    date: date

# --- Helper Functions ---
async def get_or_create_thread(user_id: str) -> str:
    if not database.is_connected:
         raise HTTPException(status_code=503, detail="Base de données non disponible.")
    try:
        query = user_threads.select().where(user_threads.c.user_id == user_id)
        result = await database.fetch_one(query)
        if result:
            logger.info(f"Thread trouvé pour user_id {user_id}: {result['thread_id']}")
            return result["thread_id"]
        else:
            logger.info(f"Aucun thread trouvé pour user_id {user_id}. Création...")
            thread = await client.beta.threads.create(metadata={"user_id": user_id})
            thread_id = thread.id
            logger.info(f"Nouveau thread créé: {thread_id}")
            insert_query = user_threads.insert().values(user_id=user_id, thread_id=thread_id)
            await database.execute(insert_query)
            logger.info(f"Nouveau thread {thread_id} associé à user_id {user_id} dans la DB.")
            return thread_id
    except Exception as e:
        logger.error(f"Erreur DB/OpenAI dans get_or_create_thread pour user_id {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la récupération/création du thread.")

async def parse_date_string(date_str: str) -> date:
    try:
        return date_parser.parse(date_str).date()
    except (ValueError, TypeError) as e:
        logger.warning(f"Erreur de parsing de date pour '{date_str}': {e}")
        raise ValueError(f"Format de date invalide: '{date_str}'. Utilisez YYYY-MM-DD ou un format similaire.")


# --- Logic métier (Async) ---
async def book_box_logic(date_obj: date, hour: str, reserved_by: str) -> bool:
    if not database.is_connected:
        logger.error("book_box_logic: Base de données non connectée.")
        return False
    try:
        query = reservations.insert().values(
            date=date_obj,
            hour=hour,
            reserved_by=reserved_by
        )
        await database.execute(query)
        logger.info(f"Réservation enregistrée: {date_obj} {hour} par {reserved_by}")
        return True
    except Exception as e:
        logger.error(f"Erreur DB dans book_box_logic: {e}")
        return False

async def report_absence_logic(name: str, date_obj: date) -> bool:
    if not database.is_connected:
        logger.error("report_absence_logic: Base de données non connectée.")
        return False
    try:
        query = absences.insert().values(
            name=name,
            date=date_obj
        )
        await database.execute(query)
        logger.info(f"Absence enregistrée: {name} le {date_obj}")
        return True
    except Exception as e:
        logger.error(f"Erreur DB dans report_absence_logic: {e}")
        return False

# --- Endpoints API ---
@app.get("/")
async def root():
    """Endpoint racine pour vérifier si l'API est en ligne."""
    return {"message": "API bot IA en ligne 👋"}

@app.post("/ask_agent")
async def ask_agent(req: Message):
    """Endpoint principal pour interroger l'assistant OpenAI."""
    logger.info(f"Requête reçue de user_id: {req.user_id}, message: '{req.message}'")
    try:
        thread_id = await get_or_create_thread(req.user_id)

        logger.info(f"Ajout du message au thread {thread_id}")
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=req.message,
            metadata={"author": req.user_id}
        )

        logger.info(f"Création du run pour le thread {thread_id} avec l'assistant {ASSISTANT_ID}")
        run = await client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID,
            metadata={"user_id": req.user_id}
        )
        logger.info(f"Run créé avec ID: {run.id}, Status: {run.status}")

        while run.status in ["queued", "in_progress", "requires_action"]:
            run = await client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            logger.debug(f"Run {run.id} Status: {run.status}")

            if run.status == "requires_action":
                logger.info(f"Run {run.id} nécessite une action (tool calls).")
                tool_outputs = []
                if run.required_action and run.required_action.type == "submit_tool_outputs":
                    for call in run.required_action.submit_tool_outputs.tool_calls:
                        func_name = call.function.name
                        arguments = json.loads(call.function.arguments)
                        logger.info(f"Exécution de l'outil: {func_name} avec args: {arguments}")
                        output = None
                        success = False
                        try:
                            if func_name == "book_box":
                                date_obj = await parse_date_string(arguments["date"])
                                success = await book_box_logic(
                                    date_obj=date_obj,
                                    hour=arguments["hour"],
                                    reserved_by=arguments.get("reserved_by", "Agent IA")
                                )
                                output = "Réservation effectuée avec succès." if success else "Échec de la réservation."
                            elif func_name == "report_absence":
                                date_obj = await parse_date_string(arguments["date"])
                                success = await report_absence_logic(
                                    name=arguments["name"],
                                    date_obj=date_obj
                                )
                                output = "Absence enregistrée avec succès." if success else "Échec de l'enregistrement de l'absence."
                            else:
                                logger.warning(f"Fonction outil non reconnue: {func_name}")
                                output = f"Outil inconnu: {func_name}"

                        except ValueError as ve:
                            output = str(ve)
                            logger.error(f"Erreur de parsing de date pour l'outil {func_name}: {ve}")
                        except Exception as e:
                            output = f"Erreur interne lors de l'exécution de l'outil {func_name}."
                            logger.error(f"Erreur inattendue pour l'outil {func_name}: {e}")

                        tool_outputs.append({"tool_call_id": call.id, "output": output})

                    logger.info(f"Soumission des sorties d'outils pour le run {run.id}")
                    run = await client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
            elif run.status in ["cancelled", "failed", "expired"]:
                logger.error(f"Le run {run.id} a échoué ou a été annulé/expiré. Status: {run.status}")
                raise HTTPException(status_code=500, detail=f"L'exécution de l'assistant a échoué (status: {run.status}).")

            await asyncio.sleep(0.5)

        logger.info(f"Run {run.id} terminé. Récupération des messages.")
        messages = await client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=1)

        if messages.data and messages.data[0].content and messages.data[0].content[0].type == 'text':
            response_text = messages.data[0].content[0].text.value
            return {
                "response": response_text,
                "thread_id": thread_id,
                "turn_id": messages.data[0].id
            }
        else:
            logger.warning(f"Aucune réponse textuelle trouvée dans le dernier message du thread {thread_id}")
            return {"response": "Je n'ai pas pu générer de réponse.", "thread_id": thread_id, "turn_id": None}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Erreur inattendue dans ask_agent pour user_id {req.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur: {e}")


@app.get("/reservations", response_model=List[ReservationOutput])
async def get_reservations():
    if not database.is_connected:
         raise HTTPException(status_code=503, detail="Base de données non disponible.")
    try:
        query = reservations.select().order_by(reservations.c.date, reservations.c.hour)
        results = await database.fetch_all(query)
        return results
    except Exception as e:
        logger.error(f"Erreur DB dans get_reservations: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des réservations.")

@app.post("/book_box", status_code=201)
async def book_box_manual(data: ReservationInput):
    try:
        date_obj = await parse_date_string(data.date)
        success = await book_box_logic(date_obj, data.hour, data.reserved_by)
        if success:
            return {"message": "Réservation ajoutée avec succès."}
        else:
            raise HTTPException(status_code=500, detail="Échec de l'enregistrement de la réservation en base de données.")
    except ValueError as ve:
         raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Erreur inattendue dans book_box_manual: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")


@app.post("/report_absence", status_code=201)
async def report_absence_manual(data: AbsenceInput):
    try:
        date_obj = await parse_date_string(data.date)
        success = await report_absence_logic(data.name, date_obj)
        if success:
            return {"message": "Absence enregistrée avec succès."}
        else:
            raise HTTPException(status_code=500, detail="Échec de l'enregistrement de l'absence en base de données.")
    except ValueError as ve:
         raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Erreur inattendue dans report_absence_manual: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")


@app.get("/absences", response_model=List[AbsenceOutput])
async def get_absences():
    if not database.is_connected:
         raise HTTPException(status_code=503, detail="Base de données non disponible.")
    try:
        query = absences.select().order_by(absences.c.date, absences.c.name)
        results = await database.fetch_all(query)
        return results
    except Exception as e:
        logger.error(f"Erreur DB dans get_absences: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des absences.")
