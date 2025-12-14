import streamlit as st
import requests
import pandas as pd
import os
from openai import OpenAI
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

st.set_page_config(page_title="Réservations des box", layout="centered")

API_URL = _get_secret("API_URL", "http://localhost:8000")

if API_URL.endswith("/ask_agent"):
    API_URL = API_URL[: -len("/ask_agent")]
API_URL = API_URL.rstrip("/")

# Debug: afficher l'URL utilisée
with st.sidebar:
    st.caption(f"🔗 API: {API_URL}")

tab1, tab2, tab3 = st.tabs([
    "📅 Réservations", 
    "🚫 Absences", 
    "📂 Vector Store"
])

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
    st.header("🤖 RAG - Base de connaissances")
    
    # Statistiques du RAG
    col1, col2, col3 = st.columns(3)
    try:
        stats_response = requests.get(f"{API_URL}/rag/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            with col1:
                st.metric("📊 Documents", stats.get("total_chunks", 0))
            with col2:
                st.metric("🧠 Embeddings", stats.get("embedding_provider", "N/A"))
            with col3:
                st.metric("💬 LLM", stats.get("llm_provider", "N/A"))
        else:
            st.info("Statistiques RAG non disponibles")
    except Exception as e:
        st.warning(f"RAG non accessible: {e}")
    
    st.divider()
    
    # Section: Poser une question au RAG
    st.subheader("❓ Poser une question")
    with st.form("rag_query"):
        query = st.text_area("Votre question", placeholder="Ex: Quelles sont les règles de réservation ?")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            top_k = st.slider("Nombre de sources", 1, 10, 5)
        with col_q2:
            use_reranker = st.checkbox("Utiliser le reranker", value=True)
        
        submit_query = st.form_submit_button("🔍 Rechercher")
        
        if submit_query and query:
            with st.spinner("Recherche en cours..."):
                try:
                    response = requests.post(
                        f"{API_URL}/rag/query",
                        json={
                            "query": query,
                            "top_k": top_k,
                            "use_reranker": use_reranker
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Réponse générée")
                        st.markdown("### 💡 Réponse")
                        st.write(result.get("answer", "Aucune réponse"))
                        
                        # Afficher les sources
                        sources = result.get("sources", [])
                        if sources:
                            st.markdown("### 📚 Sources utilisées")
                            for i, source in enumerate(sources, 1):
                                with st.expander(f"Source {i} - Score: {source.get('score', 0):.3f}"):
                                    st.text(source.get("content", ""))
                                    st.caption(f"Métadonnées: {source.get('metadata', {})}")
                    else:
                        st.error(f"Erreur: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
    
    st.divider()
    
    # Section: Upload de documents
    st.subheader("📤 Ajouter des documents")
    
    # Mode 1: Upload de fichier
    uploaded_file = st.file_uploader(
        "Uploader un fichier", 
        type=["pdf", "docx", "txt", "md"],
        help="Formats supportés: PDF, DOCX, TXT, MD"
    )
    
    if uploaded_file is not None:
        if st.button("📁 Ajouter au RAG"):
            with st.spinner("Traitement du document..."):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(
                        f"{API_URL}/rag/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result.get('message', 'Document ajouté')}")
                        st.info(f"📊 {result.get('chunk_count', 0)} chunks créés")
                        st.rerun()
                    else:
                        st.error(f"❌ Erreur: {response.text}")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
    
    # Mode 2: Texte direct
    with st.expander("✍️ Ou ajouter du texte directement"):
        with st.form("add_text_document"):
            doc_content = st.text_area("Contenu du document", height=200)
            doc_source = st.text_input("Nom du document", placeholder="Ex: reglement.txt")
            submit_text = st.form_submit_button("Ajouter")
            
            if submit_text and doc_content:
                try:
                    response = requests.post(
                        f"{API_URL}/rag/documents",
                        json={
                            "content": doc_content,
                            "source": doc_source or "document_texte"
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result.get('message', 'Document ajouté')}")
                        st.info(f"📊 {result.get('chunk_count', 0)} chunks créés")
                    else:
                        st.error(f"❌ Erreur: {response.text}")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
    
    st.divider()
    
    # Section: Gestion du vector store
    st.subheader("⚙️ Gestion")
    col_manage1, col_manage2 = st.columns(2)
    
    with col_manage1:
        if st.button("🔄 Rafraîchir les stats"):
            st.rerun()
    
    with col_manage2:
        if st.button("🗑️ Vider le vector store", type="secondary"):
            if st.session_state.get("confirm_clear"):
                try:
                    response = requests.delete(f"{API_URL}/rag/clear")
                    if response.status_code == 200:
                        st.success("✅ Vector store vidé")
                        st.session_state.pop("confirm_clear")
                        st.rerun()
                    else:
                        st.error("❌ Erreur lors du vidage")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
            else:
                st.session_state["confirm_clear"] = True
                st.warning("⚠️ Cliquez à nouveau pour confirmer")
                st.rerun()
