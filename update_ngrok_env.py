import requests
import os

ENV_PATH = ".env"

def get_ngrok_url():
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        tunnels = response.json()["tunnels"]
        for tunnel in tunnels:
            if tunnel["proto"] == "https":
                return tunnel["public_url"]
    except Exception as e:
        print("❌ Erreur : impossible de récupérer l’URL ngrok — ngrok est bien lancé ?")
        print(e)
        return None

def update_env(api_url):
    if not api_url:
        return
    lines = []
    updated = False
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r") as f:
            for line in f:
                if line.startswith("API_URL="):
                    lines.append(f"API_URL={api_url}/ask_agent\n")
                    updated = True
                else:
                    lines.append(line)
    if not updated:
        lines.append(f"API_URL={api_url}/ask_agent\n")
    with open(ENV_PATH, "w") as f:
        f.writelines(lines)
    print(f"✅ API_URL mis à jour dans {ENV_PATH} : {api_url}/ask_agent")

if __name__ == "__main__":
    url = get_ngrok_url()
    if url:
        update_env(url)
