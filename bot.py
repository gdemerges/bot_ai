import discord
from discord.ext import commands
import requests
import os
import asyncpraw
import asyncio
import httpx
import aiofiles
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
API_URL = os.getenv("API_URL")

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

@bot.event
async def on_ready():
    print(f"✅ Connecté en tant que {bot.user}")
    bot.loop.create_task(check_reddit())

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
            else:
                await message.channel.send("❌ Erreur API.")
        except Exception as e:
            await message.channel.send(f"❌ Erreur : {e}")

bot.run(TOKEN)
