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

# Configuration pour le mode de réponse
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() in {"1", "true", "yes"}
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
USE_RAG = os.getenv("USE_RAG", "false").lower() in {"1", "true", "yes"}


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


def split_message(text: str, max_length: int = 2000) -> list[str]:
    """
    Divise un message en plusieurs parties si nécessaire (limite Discord: 2000 caractères)
    """
    if len(text) <= max_length:
        return [text]
    
    parts = []
    while text:
        # Si le texte restant est plus court que la limite, l'ajouter
        if len(text) <= max_length:
            parts.append(text)
            break
        
        # Trouver le dernier espace ou saut de ligne avant la limite
        split_pos = max_length
        for sep in ['\n\n', '\n', '. ', ', ', ' ']:
            pos = text.rfind(sep, 0, max_length)
            if pos != -1:
                split_pos = pos + len(sep)
                break
        
        # Ajouter la partie et continuer avec le reste
        parts.append(text[:split_pos].rstrip())
        text = text[split_pos:].lstrip()
    
    return parts


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
    print(f"🤖 Mode LLM: {'Local (Ollama)' if USE_LOCAL_LLM else 'OpenAI API'}")
    if USE_LOCAL_LLM:
        print(f"📡 Ollama URL: {OLLAMA_BASE_URL}")
        print(f"🧠 Modèle: {OLLAMA_LLM_MODEL}")
    if USE_RAG:
        print(f"📚 RAG activé")
    print(f"🔧 API URL: {API_URL}")
    
    global reddit_task, monitoring_task
    if reddit_task is None:
        reddit_task = bot.loop.create_task(check_reddit())
    if MONITORING_ENABLED and monitoring_task is None:
        monitoring_task = bot.loop.create_task(monitor_metrics())


@bot.command(name='rag')
async def rag_command(ctx, action: str = "help"):
    """
    Commandes RAG: !rag files, !rag stats, !rag clear
    """
    if action == "files":
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{API_URL.replace('/ask_agent', '')}/rag/files")
                
                if response.status_code == 200:
                    data = response.json()
                    total = data.get("total_files", 0)
                    files = data.get("files", [])
                    
                    if total == 0:
                        await ctx.send("📂 Aucun fichier dans le RAG.")
                        return
                    
                    message = f"📂 **{total} fichier(s) indexé(s):**\n\n"
                    for file_info in files[:20]:  # Limiter à 20 fichiers
                        filename = file_info.get("filename", "Unknown")
                        chunk_count = file_info.get("chunk_count", 0)
                        message += f"• `{filename}` ({chunk_count} chunks)\n"
                    
                    if len(files) > 20:
                        message += f"\n... et {len(files) - 20} autres fichiers"
                    
                    await ctx.send(message)
                else:
                    await ctx.send("❌ Erreur lors de la récupération des fichiers")
        except Exception as e:
            await ctx.send(f"❌ Erreur: {e}")
    
    elif action == "stats":
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{API_URL.replace('/ask_agent', '')}/rag/stats")
                
                if response.status_code == 200:
                    data = response.json()
                    await ctx.send(
                        f"📊 **Statistiques RAG:**\n"
                        f"• Total chunks: {data.get('total_chunks', 0)}\n"
                        f"• Embeddings: {data.get('embedding_provider', 'N/A')}\n"
                        f"• LLM: {data.get('llm_provider', 'N/A')}\n"
                        f"• Vector Store: {data.get('vector_store_type', 'N/A')}"
                    )
                else:
                    await ctx.send("❌ Erreur lors de la récupération des stats")
        except Exception as e:
            await ctx.send(f"❌ Erreur: {e}")
    
    elif action == "clear":
        await ctx.send("⚠️ Voulez-vous vraiment vider le RAG ? Tapez `!rag confirm-clear` pour confirmer.")
    
    elif action == "confirm-clear":
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(f"{API_URL.replace('/ask_agent', '')}/rag/clear")
                
                if response.status_code == 200:
                    await ctx.send("✅ RAG vidé avec succès")
                else:
                    await ctx.send("❌ Erreur lors du vidage")
        except Exception as e:
            await ctx.send(f"❌ Erreur: {e}")
    
    else:
        await ctx.send(
            "📚 **Commandes RAG disponibles:**\n"
            "• `!rag files` - Liste les fichiers indexés\n"
            "• `!rag stats` - Affiche les statistiques\n"
            "• `!rag clear` - Vide le vector store"
        )


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
        await message.channel.send("📁 Fichier reçu. Ajout au RAG en cours...")
        
        for attachment in message.attachments:
            # Vérifier l'extension du fichier
            filename = attachment.filename.lower()
            supported_extensions = ['.pdf', '.txt', '.md', '.docx']
            
            if not any(filename.endswith(ext) for ext in supported_extensions):
                await message.channel.send(f"⚠️ Type de fichier non supporté: {attachment.filename}\nFormats acceptés: PDF, TXT, MD, DOCX")
                continue
            
            file_path = f"/tmp/{attachment.filename}"
            
            try:
                # Télécharger le fichier
                await attachment.save(file_path)
                
                if USE_RAG:
                    # Envoyer au RAG local
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        with open(file_path, "rb") as f:
                            files = {"file": (attachment.filename, f, "application/octet-stream")}
                            response = await client.post(
                                f"{API_URL.replace('/ask_agent', '')}/rag/upload",
                                files=files
                            )
                        
                        if response.status_code == 200:
                            data = response.json()
                            chunk_count = data.get("chunk_count", 0)
                            await message.channel.send(f"✅ {attachment.filename} ajouté au RAG ({chunk_count} chunks créés)")
                        else:
                            error = response.json().get("detail", "Erreur inconnue")
                            await message.channel.send(f"❌ Erreur RAG: {error}")
                else:
                    # Fallback sur OpenAI Vector Store
                    client_openai = OpenAI()
                    vectorstore_id = os.getenv("OPENAI_VECTORSTORE_ID")
                    with open(file_path, "rb") as f:
                        client_openai.vector_stores.file_batches.upload_and_poll(
                            vector_store_id=vectorstore_id,
                            files=[f]
                        )
                    await message.channel.send(f"✅ {attachment.filename} ajouté au vector store OpenAI")
                    
            except Exception as e:
                print(f"❌ Erreur fichier: {e}")
                import traceback
                traceback.print_exc()
                await message.channel.send(f"❌ Erreur lors de l'ajout du fichier : {e}")
            finally:
                os.remove(file_path)
        return

    question = message.content.replace(bot.user.mention, "").strip() if not is_dm else message.content.strip()
    if not question:
        await message.channel.send("Tu dois poser une question.")
        return

    async with message.channel.typing():
        # Récupérer l'historique
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
            # Choisir entre LLM local ou API OpenAI
            if USE_LOCAL_LLM:
                print(f"🤖 Génération locale pour: {question[:50]}...")
                result = await generate_local_response(question, historique)
            else:
                print(f"🌐 Utilisation de l'API OpenAI pour: {question[:50]}...")
                # Utiliser l'API existante (OpenAI Assistant)
                payload = {
                    "message": question,
                    "user_id": str(message.author.id),
                    "history": historique
                }
                async with httpx.AsyncClient(timeout=100.0) as http_client:
                    response = await http_client.post(API_URL, json=payload)

                if response.status_code == 200:
                    result = response.json().get("response", "Aucune réponse.")
                elif response.status_code == 503:
                    result = "Service indisponible : base de données hors ligne"
                else:
                    result = "❌ Erreur API."
            
            # Diviser et envoyer la réponse si elle est trop longue
            message_parts = split_message(result)
            
            for i, part in enumerate(message_parts):
                if i == 0:
                    # Premier message : reply ou send selon le contexte
                    if is_dm:
                        await message.channel.send(part)
                    else:
                        await message.reply(part)
                else:
                    # Messages suivants : toujours send
                    await message.channel.send(part)
                    # Petite pause pour éviter le rate limiting
                    await asyncio.sleep(0.5)
                
        except Exception as e:
            print(f"❌ Erreur complète: {e}")
            import traceback
            traceback.print_exc()
            await message.channel.send(f"❌ Erreur : {e}")


async def generate_local_response(question: str, historique: list) -> str:
    """
    Génère une réponse en utilisant Ollama (local) ou le RAG
    """
    try:
        print(f"🔍 USE_RAG = {USE_RAG}")
        
        # Si le RAG est activé, utiliser l'endpoint RAG
        if USE_RAG:
            print(f"🔍 Recherche RAG pour: {question[:100]}")
            async with httpx.AsyncClient(timeout=120.0) as client:
                
                # Requête RAG - utiliser uniquement la question pour une meilleure précision
                response = await client.post(
                    f"{API_URL.replace('/ask_agent', '')}/rag/query",
                    json={
                        "query": question,
                        "top_k": 10,  # Récupérer plus de documents
                        "use_reranker": True  # Activer le reranking pour améliorer la pertinence
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Aucune réponse.")
                    sources = data.get("sources", [])
                    
                    # Log pour déboguer
                    print(f"✅ RAG: {len(sources)} sources trouvées")
                    if sources:
                        print(f"📄 Meilleure source (score: {sources[0].get('score', 'N/A')}): {sources[0].get('metadata', {}).get('source', 'unknown')}")
                    
                    return answer
                else:
                    print(f"⚠️ RAG erreur {response.status_code}, fallback sur Ollama")
                    # Fallback sur Ollama sans RAG
                    return await generate_ollama_response(question, historique)
        else:
            # Utiliser directement Ollama
            return await generate_ollama_response(question, historique)
            
    except Exception as e:
        print(f"Erreur génération locale: {e}")
        return f"Erreur lors de la génération de la réponse: {e}"


async def generate_ollama_response(question: str, historique: list) -> str:
    """
    Génère une réponse avec Ollama directement
    """
    # Construire les messages pour Ollama
    messages = []
    
    # Ajouter le prompt système
    system_prompt = """Tu es un assistant Discord utile et amical. 
Réponds de manière concise et naturelle aux questions.
Adapte ton ton à la conversation Discord."""
    
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    # Ajouter l'historique (derniers 10 messages)
    for msg in historique[-10:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Ajouter la question actuelle si elle n'est pas déjà dans l'historique
    if not historique or historique[-1]["content"] != question:
        messages.append({
            "role": "user",
            "content": question
        })
    
    # Appeler Ollama
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_LLM_MODEL,
                "messages": messages,
                "stream": False
            }
        )
        
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "Aucune réponse.")


bot.run(TOKEN)
