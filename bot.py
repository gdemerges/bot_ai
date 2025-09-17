import asyncio
import os
import re
from typing import Optional

import asyncpraw
import discord
import httpx
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
API_URL = os.getenv("API_URL")
MONITORING_ENABLED = os.getenv("MONITORING_ENABLED", "false").lower() in {"1", "true", "yes"}
MONITORING_INTERVAL = int(os.getenv("MONITORING_INTERVAL", "60"))
MONITORING_FAILURE_THRESHOLD = int(os.getenv("MONITORING_FAILURE_THRESHOLD", "3"))


def _parse_int(env_value: Optional[str]) -> Optional[int]:
    if env_value is None:
        return None
    try:
        return int(env_value)
    except (TypeError, ValueError):
        print(f"Valeur d'identifiant invalide pour le monitoring : {env_value}")
        return None


def _derive_metrics_url(api_url: Optional[str]) -> Optional[str]:
    if not api_url:
        return None
    base = api_url.rstrip("/")
    if base.endswith("/ask_agent"):
        base = base[: -len("/ask_agent")]
    return f"{base}/metrics"


MONITORING_METRICS_URL = os.getenv("MONITORING_METRICS_URL") or _derive_metrics_url(API_URL)
MONITORING_CHANNEL_ID = _parse_int(os.getenv("MONITORING_CHANNEL_ID"))
MONITORING_USER_ID = _parse_int(os.getenv("MONITORING_USER_ID"))

monitoring_channel = None
monitoring_user = None
monitoring_task = None
reddit_task = None

if not TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN non défini !")

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


subreddit_name = "Hellfest"

SEEN_FILE = "seen_posts.txt"
seen_posts = set()

def load_seen_posts():
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, "r") as f:
            for line in f:
                seen_posts.add(line.strip())

def save_seen_post(post_id):
    with open(SEEN_FILE, "a") as f:
        f.write(f"{post_id}\n")

async def check_reddit():
    await bot.wait_until_ready()
    load_seen_posts()
    user = await bot.fetch_user(282150973810540566)

    while not bot.is_closed():
        try:
            reddit = asyncpraw.Reddit(
                    client_id=os.getenv("REDDIT_CLIENT_ID"),
                    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                    user_agent="hellfest/1.0 (by u/Sagi1308)"
                )
            async with reddit:
                subreddit = await reddit.subreddit(subreddit_name)
                async for submission in subreddit.new(limit=3):
                    if submission.id not in seen_posts:
                        message = f"🚨 Nouveau post sur r/{subreddit_name} : **{submission.title}**\n🔗 {submission.url}"
                        await user.send(message)
                        seen_posts.add(submission.id)
                        save_seen_post(submission.id)
        except (asyncpraw.exceptions.PRAWException, httpx.RequestError) as e:
            print(f"Erreur Reddit : {e}")
        interval = int(os.getenv("REDDIT_INTERVAL", "60"))
        await asyncio.sleep(interval)


async def _send_monitoring_alert(message: str) -> None:
    global monitoring_channel, monitoring_user

    if MONITORING_CHANNEL_ID:
        if monitoring_channel is None:
            monitoring_channel = bot.get_channel(MONITORING_CHANNEL_ID)
            if monitoring_channel is None:
                try:
                    monitoring_channel = await bot.fetch_channel(MONITORING_CHANNEL_ID)
                except Exception as channel_error:
                    print(f"Impossible de récupérer le canal de monitoring : {channel_error}")
                    monitoring_channel = None
        if monitoring_channel is not None:
            try:
                await monitoring_channel.send(message)
                return
            except Exception as send_error:
                print(f"Erreur lors de l'envoi de l'alerte sur le canal : {send_error}")

    if MONITORING_USER_ID:
        if monitoring_user is None:
            try:
                monitoring_user = await bot.fetch_user(MONITORING_USER_ID)
            except Exception as user_error:
                print(f"Impossible de récupérer l'utilisateur de monitoring : {user_error}")
                monitoring_user = None
        if monitoring_user is not None:
            try:
                await monitoring_user.send(message)
                return
            except Exception as dm_error:
                print(f"Erreur lors de l'envoi de l'alerte en DM : {dm_error}")

    print(f"[MONITORING] {message}")


async def monitor_metrics():
    if not MONITORING_METRICS_URL:
        print("Monitoring activé mais aucune URL de métriques n'est configurée.")
        return

    await bot.wait_until_ready()

    failure_streak = 0
    alert_active = False
    last_5xx_count: Optional[float] = None

    async with httpx.AsyncClient(timeout=10.0) as client:
        while not bot.is_closed():
            try:
                response = await client.get(MONITORING_METRICS_URL)
                response.raise_for_status()
                metrics_text = response.text

                if alert_active:
                    await _send_monitoring_alert("✅ Les métriques de l'API sont de nouveau accessibles.")
                    alert_active = False

                failure_streak = 0

                match = re.search(
                    r'^http_requests_total\{[^}]*status="5xx"[^}]*\}\s+([0-9eE+\-\.]+)',
                    metrics_text,
                    re.MULTILINE,
                )

                if match:
                    current_5xx = float(match.group(1))
                    if last_5xx_count is not None and current_5xx > last_5xx_count:
                        delta = int(current_5xx - last_5xx_count)
                        await _send_monitoring_alert(
                            f"🚨 {delta} nouvelle(s) réponse(s) 5xx détectée(s) sur l'API."
                        )
                    last_5xx_count = current_5xx

            except Exception as error:
                failure_streak += 1
                if failure_streak >= MONITORING_FAILURE_THRESHOLD and not alert_active:
                    await _send_monitoring_alert(
                        f"🚨 Impossible d'accéder aux métriques ({failure_streak} tentative(s)) : {error}"
                    )
                    alert_active = True

            await asyncio.sleep(MONITORING_INTERVAL)


@bot.event
async def on_ready():
    print(f"✅ Connecté en tant que {bot.user}")
    global reddit_task, monitoring_task
    if reddit_task is None:
        reddit_task = bot.loop.create_task(check_reddit())
    if MONITORING_ENABLED and monitoring_task is None:
        monitoring_task = bot.loop.create_task(monitor_metrics())

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    is_dm = message.guild is None
    is_mention = bot.user.mention in message.content
    is_kim = str(message.author.id) == "1192414156243091609"
    should_respond = is_dm or is_mention or is_kim

    if not should_respond:
        return

    if message.attachments:
        await message.channel.send("📁 Fichier reçu. Traitement en cours...")
        for attachment in message.attachments:
            file_path = f"/tmp/{attachment.filename}"
            await attachment.save(file_path)

            try:
                client = OpenAI()
                vectorstore_id = os.getenv("OPENAI_VECTORSTORE_ID")
                with open(file_path, "rb") as f:
                    client.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=vectorstore_id,
                        files=[f]
                    )
                await message.channel.send(f"✅ Fichier ajouté au vector store : {attachment.filename}")
            except Exception as e:
                await message.channel.send(f"❌ Erreur lors de l'ajout du fichier : {e}")
            finally:
                os.remove(file_path)
        return

    question = message.content.replace(bot.user.mention, "").strip() if not is_dm else message.content.strip()
    if not question:
        await message.channel.send("Tu dois poser une question.")
        return

    async with message.channel.typing():
        historique = []
        async for msg in message.channel.history(limit=15, oldest_first=True):
            if not msg.content:
                continue
            role = "assistant" if msg.author == bot.user else "user"
            historique.append({
                "author": msg.author.name,
                "content": msg.content,
                "role": role
            })

        try:
            payload = {
                "message": question,
                "user_id": str(message.author.id),
                "history": historique
            }
            async with httpx.AsyncClient(timeout=100.0) as client:
                response = await client.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json().get("response", "Aucune réponse.")
                if is_dm:
                    await message.channel.send(result)
                else:
                    await message.reply(result)
            elif response.status_code == 503:
                await message.channel.send("Service indisponible : base de données hors ligne")
            else:
                await message.channel.send("❌ Erreur API.")
        except Exception as e:
            await message.channel.send(f"❌ Erreur : {e}")

bot.run(TOKEN)
