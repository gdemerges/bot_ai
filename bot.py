import discord
from discord.ext import commands
import requests
import os
import asyncpraw
import asyncio
import httpx
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
API_URL = "http://localhost:8000/ask_agent"

if not TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN non défini !")

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

subreddit_name = "Hellfest"
seen_posts = set()

async def check_reddit():
    """Tâche de fond qui vérifie les nouveaux posts sur r/Hellfest"""
    await bot.wait_until_ready()
    user = await bot.fetch_user(282150973810540566)

    try:
        reddit = asyncpraw.Reddit(
            client_id="Wv9IGH0gvCczyoh9n49tlg",
            client_secret="z2mqyGPAhCkBGnZI9Q8q9ZCgr3Kw2Q",
            user_agent="hellfest/1.0 (by u/Sagi1308)"
        )
        async with reddit:
            subreddit = await reddit.subreddit(subreddit_name)
            async for submission in subreddit.new(limit=3):
                if submission.id not in seen_posts:
                    message = f"🚨 Nouveau post sur r/{subreddit_name} : **{submission.title}**\n🔗 {submission.url}"
                    await user.send(message)
                    seen_posts.add(submission.id)
            await asyncio.sleep(60)
    except (asyncpraw.exceptions.PRAWException, httpx.RequestError) as e:
        print(f"Erreur Reddit : {e}")
        await asyncio.sleep(60)

@bot.event
async def on_ready():
    print(f"✅ Connecté en tant que {bot.user}")
    bot.loop.create_task(check_reddit())

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    if (str(message.author.id) == "1192414156243091609" and
        message.guild and str(message.guild.id) == "1200438506762293300"):
        try:
            await message.add_reaction("❤️")
        except discord.HTTPException as e:
            print(f"Erreur lors de l'ajout de la réaction : {e}")

    is_dm = message.guild is None
    is_mention = bot.user.mention in message.content

    if is_dm or is_mention:
        if not is_dm:
            question = message.content.replace(bot.user.mention, "").strip()
        else:
            question = message.content.strip()

        if not question:
            await message.channel.send("Tu dois poser une question.")
            return

        async with message.channel.typing():
            try:
                response = requests.post(API_URL, json={
                    "message": question,
                    "user_id": str(message.author.id) 
                }, timeout=20)
                if response.status_code == 200:
                    result = response.json().get("response", "Aucune réponse.")
                    await message.channel.send(f"{result}")
                else:
                    await message.channel.send("❌ Erreur API.")
            except Exception as e:
                await message.channel.send(f"❌ Erreur : {e}")

bot.run(TOKEN)
