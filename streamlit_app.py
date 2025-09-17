import streamlit as st
import requests
import pandas as pd
import os
import openai
from openai import OpenAI
import jwt
import datetime
from dotenv import load_dotenv

load_dotenv()
# --- Secrets helper (works on Streamlit Cloud & locally) ---
def _get_secret(key: str, default: str | None = None):
    try:
        if hasattr(st, "secrets") and st.secrets is not None:
            val = st.secrets.get(key)
            if val is not None:
                return val
    except Exception:
        pass
    return os.getenv(key, default)

JWT_SECRET = _get_secret("JWT_SECRET")

st.set_page_config(page_title="Réservations des box", layout="centered")

API_URL = _get_secret("API_URL", "http://localhost:8000")

if API_URL.endswith("/ask_agent"):
    API_URL = API_URL[: -len("/ask_agent")]
API_URL = API_URL.rstrip("/")

if _get_secret("DEBUG_AUTH", "false").lower() == "true":
    with st.sidebar.expander("🔎 Debug auth (masqué en prod)"):
        st.write({
            "JWT_SECRET_set": bool(JWT_SECRET),
            "AUTH_USERNAME_set": bool(_get_secret("AUTH_USERNAME")),
            "AUTH_PASSWORD_set": bool(_get_secret("AUTH_PASSWORD")),
            "API_URL": API_URL,
        })

tab1, tab2, tab3 = st.tabs([
    "📅 Réservations", 
    "🚫 Absences", 
    "📂 Vector Store"
])

def generate_token(username: str):
    payload = {
        "user": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload["user"]
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

def login_form():
    st.sidebar.title("🔐 Connexion")
    username = st.sidebar.text_input("Nom d'utilisateur")
    password = st.sidebar.text_input("Mot de passe", type="password")
    login_btn = st.sidebar.button("Se connecter")

    if login_btn:
        valid_username = _get_secret("AUTH_USERNAME")
        valid_password = _get_secret("AUTH_PASSWORD")
        if username == valid_username and password == valid_password:
            token = generate_token(username)
            st.session_state["auth_token"] = token
            st.success("Connexion réussie ✅")
            st.rerun()
        else:
            st.error("Identifiants invalides ❌")

def check_auth():
    token = st.session_state.get("auth_token")
    if token:
        user = verify_token(token)
        if user:
            return True
    return False

if not JWT_SECRET:
    st.sidebar.error("JWT_SECRET manquant : définissez-le dans Settings → Secrets sur Streamlit Cloud.")

if not check_auth():
    login_form()
    st.stop()

with tab1:
    st.header("➕ Ajouter une réservation")
    with st.form("add_reservation"):
        date = st.date_input("Date")
        hour = st.time_input("Heure")
        reserved_by = st.text_input("Réservé par", value="")
        submitted = st.form_submit_button("Réserver")

        if submitted:
            payload = {
                "date": str(date),
                "hour": hour.strftime("%H:%M"),
                "reserved_by": reserved_by or "Agent"
            }
            r = requests.post(f"{API_URL}/book_box", json=payload)
            if r.status_code == 200:
                st.success("Réservation ajoutée avec succès ✅")
            else:
                st.error("Erreur lors de la réservation ❌")

    st.header("🗓️ Réservations actuelles")
    response = requests.get(f"{API_URL}/reservations")
    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values(by=["date", "hour"])
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": st.column_config.DatetimeColumn("📅 Date", format="YYYY-MM-DD"),
                    "hour": st.column_config.TextColumn("🕒 Heure"),
                    "reserved_by": st.column_config.TextColumn("👤 Réservé par"),
                },
            )
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Export CSV", csv, "reservations.csv", "text/csv")
        else:
            st.info("Aucune réservation pour le moment.")
    else:
        st.error("Erreur lors de la récupération des réservations.")

with tab2:
    st.header("🚫 Déclarer une absence")
    with st.form("report_absence"):
        name = st.text_input("Nom de l'apprenant")
        absence_date = st.date_input("Date de l'absence")
        submit_abs = st.form_submit_button("Enregistrer l'absence")

        if submit_abs:
            payload = {"name": name, "date": str(absence_date)}
            r = requests.post(f"{API_URL}/report_absence", json=payload)
            if r.status_code == 200:
                st.success("Absence enregistrée ✅")
            else:
                st.error("Erreur lors de l'enregistrement ❌")

    st.header("📋 Absences enregistrées")
    response_abs = requests.get(f"{API_URL}/list_absences")
    if response_abs.status_code == 200:
        abs_data = response_abs.json()
        if abs_data:
            df_abs = pd.DataFrame(abs_data)

            # Normaliser les noms de colonnes possibles pour la date
            if 'date' not in df_abs.columns:
                if 'absence_date' in df_abs.columns:
                    df_abs['date'] = df_abs['absence_date']
                elif 'created_at' in df_abs.columns:
                    df_abs['date'] = df_abs['created_at']
                elif 'day' in df_abs.columns:
                    df_abs['date'] = df_abs['day']
                else:
                    st.warning("Le champ 'date' est absent de la réponse de l'API. Affichage brut des données.")
                    st.dataframe(df_abs)
                    st.stop()

            # Normaliser le nom de l'apprenant
            if 'name' not in df_abs.columns:
                if 'student' in df_abs.columns:
                    df_abs['name'] = df_abs['student']
                elif 'reserved_by' in df_abs.columns:
                    # fallback très permissif si l'API renvoie un autre libellé
                    df_abs['name'] = df_abs['reserved_by']

            # Conversion de la colonne date et tri
            df_abs['date'] = pd.to_datetime(df_abs['date'], errors='coerce', utc=True).dt.tz_convert(None)
            sort_cols = [c for c in ["date", "name"] if c in df_abs.columns]
            if sort_cols:
                df_abs = df_abs.sort_values(by=sort_cols)

            st.dataframe(
                df_abs,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": st.column_config.DatetimeColumn("📅 Date", format="YYYY-MM-DD"),
                    "name": st.column_config.TextColumn("👤 Nom"),
                },
            )
            csv_abs = df_abs.to_csv(index=False).encode()
            st.download_button("⬇️ Export CSV", csv_abs, "absences.csv", "text/csv", key="dl_abs_csv")
        else:
            st.info("Aucune absence enregistrée pour le moment.")
    else:
        st.error("Erreur lors de la récupération des absences.")

with tab3:
    st.header("📄 Ajouter un document à la base de connaissances")
    uploaded_file = st.file_uploader("Choisir un fichier", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        st.info("📤 Téléversement en cours...")
        try:
            client = OpenAI()

            with open(f"/tmp/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())

            vectorstore_id = _get_secret("OPENAI_VECTORSTORE_ID")
            with open(f"/tmp/{uploaded_file.name}", "rb") as f:
                client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vectorstore_id,
                    files=[f]
                )
            st.success(f"✅ Fichier '{uploaded_file.name}' ajouté au vector store.")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'ajout : {e}")

    st.header("📂 Fichiers dans le vector store")
    try:
        vectorstore_id = _get_secret("OPENAI_VECTORSTORE_ID")
        client = OpenAI()
        file_list = client.vector_stores.files.list(vector_store_id=vectorstore_id)
        files = list(file_list)

        if files:
            file_data = []
            for f in files:
                full_file = client.files.retrieve(f.id)
                file_data.append({
                    "Nom du fichier": full_file.filename,
                    "ID": full_file.id,
                    "Statut": f.status
                })
            df_files = pd.DataFrame(file_data)
            st.dataframe(df_files)
        else:
            st.info("Aucun fichier actuellement dans le vector store.")
    except Exception as e:
        st.error(f"❌ Erreur lors de la récupération des fichiers : {e}")

if st.sidebar.button("🚪 Se déconnecter"):
    st.session_state.pop("auth_token", None)
    st.rerun()
