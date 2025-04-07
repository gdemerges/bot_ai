from dotenv import load_dotenv
import os

env = os.getenv("ENV", "local")
env_path = f".env.{env}"
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv() 
