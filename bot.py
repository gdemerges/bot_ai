import discord
from discord.ext import commands, tasks
import os
import asyncio
import logging
import httpx
import aiofiles
import asyncpraw
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
API_URL = os.getenv("API_URL", "http://api:8000/ask_agent")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "discord-bot/1.0 (by u/YourRedditUsername)")
REDDIT_SUBREDDIT = os.getenv("REDDIT_SUBREDDIT", "Hellfest")
REDDIT_INTERVAL = int(os.getenv("REDDIT_INTERVAL", "300"))
REDDIT_NOTIFY_USER_ID = int(os.getenv("REDDIT_NOTIFY_USER_ID"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_VECTORSTORE_ID = os.getenv("OPENAI_VECTORSTORE_ID")

if not TOKEN:
    raise ValueError("Le DISCORD_BOT_TOKEN n'est pas défini dans le .env !")
if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
    logger.warning("Credentials Reddit manquants, la surveillance Reddit sera désactivée.")
    REDDIT_ENABLED = False
else:
    REDDIT_ENABLED = True
if not REDDIT_NOTIFY_USER_ID and REDDIT_ENABLED:
    logger.warning("REDDIT_NOTIFY_USER_ID manquant, notifications Reddit désactivées.")
    REDDIT_ENABLED = False
if not OPENAI_API_KEY or not OPENAI_VECTORSTORE_ID:
    logger.warning("Configuration OpenAI (API Key ou Vector Store ID) manquante, l'upload de fichiers sera désactivé.")
    FILE_UPLOAD_ENABLED = False
else:
    FILE_UPLOAD_ENABLED = True

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

if FILE_UPLOAD_ENABLED:
    aclient_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

SEEN_FILE = "seen_posts.txt"
seen_posts = set()

async def load_seen_posts():
    seen_posts.clear()
    if os.path.exists(SEEN_FILE):
        try:
            async with aiofiles.open(SEEN_FILE, "r") as f:
                async for line in f:
                    seen_posts.add(line.strip())
            logger.info(f"{len(seen_posts)} posts vus chargés depuis {SEEN_FILE}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {SEEN_FILE}: {e}")

async def save_seen_post(post_id):
    try:
        async with aiofiles.open(SEEN_FILE, "a") as f:
            await f.write(f"{post_id}\n")
        logger.debug(f"Post {post_id} ajouté à {SEEN_FILE}")
    except Exception as e:
        logger.error(f"Erreur lors de l'écriture dans {SEEN_FILE}: {e}")

@tasks.loop(seconds=REDDIT_INTERVAL)
async def check_reddit_task():
    if not REDDIT_ENABLED:
        return

    notify_user = bot.get_user(REDDIT_NOTIFY_USER_ID)
    if not notify_user:
        logger.warning(f"Impossible de trouver l'utilisateur Discord avec ID {REDDIT_NOTIFY_USER_ID} pour les notifications Reddit.")
        return

    reddit = None
    try:
        reddit = asyncpraw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
        async with reddit:
            subreddit = await reddit.subreddit(REDDIT_SUBREDDIT)
            async for submission in subreddit.new(limit=5):
                if submission.id not in seen_posts:
                    logger.info(f"Nouveau post trouvé sur Reddit: {submission.id} - {submission.title}")
                    message = (
                        f"🚨 Nouveau post sur r/{REDDIT_SUBREDDIT} : **{submission.title}**\n"
                        f"🔗 {submission.url}"
                    )
                    try:
                        await notify_user.send(message)
                        seen_posts.add(submission.id)
                        await save_seen_post(submission.id)
                    except discord.errors.Forbidden:
                        logger.error(f"Impossible d'envoyer un DM à l'utilisateur {notify_user.name} (ID: {REDDIT_NOTIFY_USER_ID}). Vérifiez les permissions/blocages.")
                    except Exception as e:
                        logger.error(f"Erreur lors de l'envoi du DM pour le post {submission.id}: {e}")

    except (asyncpraw.exceptions.PRAWException, httpx.RequestError, httpx.TimeoutException) as e:
        logger.error(f"Erreur lors de la connexion/requête à Reddit: {e}")
    except Exception as e:
        logger.exception(f"Erreur inattendue dans la tâche Reddit: {e}")

@check_reddit_task.before_loop
async def before_check_reddit():
    await bot.wait_until_ready()
    await load_seen_posts()

@bot.event
async def on_ready():
    logger.info(f"✅ Connecté en tant que {bot.user}")
    if REDDIT_ENABLED:
        check_reddit_task.start()

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mention = bot.user.mentioned_in(message)

    if is_dm or is_mention:

        if message.attachments and FILE_UPLOAD_ENABLED:
            if not OPENAI_VECTORSTORE_ID:
                 await message.channel.send("❌ L'upload de fichiers est configuré mais l'ID du Vector Store est manquant.")
                 return

            await message.channel.send(f"📁 {len(message.attachments)} fichier(s) reçu(s). Tentative d'ajout au Vector Store...")

            files_to_upload = []
            saved_files_paths = []

            for attachment in message.attachments:
                file_path = f"/tmp/{attachment.filename}"
                try:
                    await attachment.save(file_path)
                    saved_files_paths.append(file_path)
                    file_handle = await aiofiles.open(file_path, "rb")
                    files_to_upload.append(file_handle)
                    logger.info(f"Fichier {attachment.filename} sauvegardé sur {file_path} et prêt pour l'upload.")

                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde du fichier {attachment.filename}: {e}")
                    await message.channel.send(f"❌ Erreur lors de la sauvegarde du fichier : {attachment.filename}")
                    for f in files_to_upload: await f.close()
                    for p in saved_files_paths: os.remove(p)
                    return

            if files_to_upload:
                try:
                    logger.info(f"Upload de {len(files_to_upload)} fichier(s) vers le Vector Store {OPENAI_VECTORSTORE_ID}...")
                    file_batch = await aclient_openai.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=OPENAI_VECTORSTORE_ID,
                        files=files_to_upload
                    )
                    successful_files = file_batch.file_counts.completed
                    failed_files = file_batch.file_counts.failed
                    logger.info(f"Upload terminé. Statut Batch: {file_batch.status}. Fichiers: {successful_files} succès, {failed_files} échecs.")

                    if failed_files > 0:
                         await message.channel.send(f"⚠️ {successful_files}/{len(files_to_upload)} fichier(s) ajouté(s) au Vector Store. {failed_files} ont échoué.")
                    else:
                         await message.channel.send(f"✅ {successful_files} fichier(s) ajouté(s) avec succès au Vector Store.")

                except Exception as e:
                    logger.exception(f"Erreur lors de l'upload OpenAI: {e}")
                    await message.channel.send(f"❌ Erreur lors de l'ajout des fichiers au Vector Store : {e}")
                finally:
                    for f in files_to_upload:
                        await f.close()
                    for file_path in saved_files_paths:
                        try:
                            os.remove(file_path)
                        except OSError as e:
                            logger.warning(f"Impossible de supprimer le fichier temporaire {file_path}: {e}")
            return

        if is_mention:
            question = message.content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()
        else:
            question = message.content.strip()

        if not question:
            if is_mention:
                 await message.channel.send("Il faut me poser une question après m'avoir mentionné !")
            return

        async with message.channel.typing():
            logger.info(f"Traitement de la question de {message.author.name} (ID: {message.author.id}): '{question}'")
            payload = {
                "message": question,
                "user_id": str(message.author.id),
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    logger.debug(f"Envoi de la requête à {API_URL} avec payload: {payload}")
                    response = await client.post(API_URL, json=payload)
                    response.raise_for_status()

                    api_data = response.json()
                    result = api_data.get("response", "Désolé, je n'ai pas reçu de réponse claire de l'API.")
                    logger.info(f"Réponse reçue de l'API pour {message.author.id}: '{result[:100]}...'")
                    await message.channel.send(result)

                except httpx.HTTPStatusError as e:
                    error_message = f"❌ Erreur de l'API ({e.response.status_code})."
                    try:
                        error_detail = e.response.json().get("detail", e.response.text)
                        error_message += f" Détail: {error_detail}"
                    except Exception: # Ignorer si le corps n'est pas du JSON ou illisible
                        pass
                    logger.error(f"Erreur HTTP lors de l'appel API pour {message.author.id}: {error_message}", exc_info=True)
                    await message.channel.send(error_message)

                except httpx.RequestError as e:
                    logger.error(f"Erreur de connexion à l'API ({API_URL}) pour {message.author.id}: {e}", exc_info=True)
                    await message.channel.send(f"❌ Impossible de contacter mon API interne. L'erreur est : {e}")

                except Exception as e:
                    logger.exception(f"Erreur inattendue lors du traitement du message de {message.author.id}: {e}")
                    await message.channel.send(f"❌ Une erreur interne est survenue lors du traitement de votre demande.")

if __name__ == "__main__":
    if not TOKEN:
        print("ERREUR: DISCORD_BOT_TOKEN n'est pas défini !")
    else:
        logger.info("Démarrage du bot Discord...")
        try:
            bot.run(TOKEN)
        except discord.errors.LoginFailure:
            logger.critical("Échec de la connexion à Discord: Token invalide.")
        except Exception as e:
            logger.critical(f"Erreur critique lors du démarrage ou de l'exécution du bot: {e}", exc_info=True)
