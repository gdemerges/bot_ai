import aiohttp
import asyncio
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = "1200438507315920918"
API_BASE = "https://discord.com/api/v10"

HEADERS = {
    "Authorization": f"Bot {DISCORD_TOKEN}"
}

async def fetch_messages(channel_id):
    messages = []
    url = f"{API_BASE}/channels/{channel_id}/messages"
    params = {"limit": 100}
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        while True:
            async with session.get(
                url,
                headers=HEADERS,
                params=params,
            ) as resp:
                if resp.status == 429:
                    data = await resp.json()
                    retry_after = data.get("retry_after", 1)
                    print(f"Rate limited. Retrying after {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                    continue
                elif resp.status != 200:
                    print(f"Erreur lors de la récupération des messages : {resp.status}")
                    break
                batch = await resp.json()
                if not batch:
                    break
                for msg in batch:
                    messages.append({
                        "user_id": msg["author"]["id"],
                        "username": msg["author"]["username"],
                        "display_name": msg["author"].get("global_name", ""), 
                        "timestamp": msg["timestamp"],
                        "content": msg["content"]
                    })
                params["before"] = batch[-1]["id"]
    return messages

async def main():
    all_messages = await fetch_messages(CHANNEL_ID)
    
    for msg in all_messages:
        msg["timestamp"] = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00")).isoformat()

    with open("messages.json", "w", encoding="utf-8") as f:
        json.dump(all_messages, f, indent=2, ensure_ascii=False)
    print(f"{len(all_messages)} messages enregistrés dans messages.json")

if __name__ == "__main__":
    asyncio.run(main())
