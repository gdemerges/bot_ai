import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Réservations des box", layout="centered")
st.title("📅 Gestion des réservations et absences")

# Ajouter une réservation
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
            
# Affichage des réservations
st.header("🗓️ Réservations actuelles")
response = requests.get(f"{API_URL}/reservations")
if response.status_code == 200:
    data = response.json()
    if data:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=["date", "hour"])
        st.dataframe(df)
    else:
        st.info("Aucune réservation pour le moment.")
else:
    st.error("Erreur lors de la récupération des réservations.")

# Ajouter une absence
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

# Affichage des absences enregistrées
st.header("📋 Absences enregistrées")
response_abs = requests.get(f"{API_URL}/absences")
if response_abs.status_code == 200:
    abs_data = response_abs.json()
    if abs_data:
        df_abs = pd.DataFrame(abs_data)
        df_abs['date'] = pd.to_datetime(df_abs['date'])
        df_abs = df_abs.sort_values(by=["date", "name"])
        st.dataframe(df_abs)
    else:
        st.info("Aucune absence enregistrée pour le moment.")
else:
    st.error("Erreur lors de la récupération des absences.")
