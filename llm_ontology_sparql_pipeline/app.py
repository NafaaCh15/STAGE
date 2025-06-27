# ==============================================================================
# FICHIER PRINCIPAL DE L'APPLICATION STREAMLIT : app.py (Version "PRO")
# Rôle : Chef d'orchestre du pipeline avec une interface utilisateur stylisée.
# ==============================================================================

# --- Imports des bibliothèques ---
import streamlit as st
import os
import logging
import io
from rdflib import Graph

# --- Imports des modules du projet ---
try:
    from src.llm.llm_response_generator import get_llm_direct_response, generate_response_from_reasoning_path
    from src.ontology.ontology_retriever import retrieve_relevant_facts
    from src.llm.llm_enriched_prompt_generator import generate_enriched_prompt_response
    from src.ontology.graph_interrogator import find_reasoning_path
except ImportError as e:
    st.error(f"Erreur d'importation des modules : {e}")
    st.stop()

# --- Configuration du Logger ---
log_stream = io.StringIO()
pipeline_logger = logging.getLogger("pipeline_trace")
if not pipeline_logger.handlers:
    pipeline_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    pipeline_logger.addHandler(handler)

# --- Configuration de la Page et Style ---
st.set_page_config(layout="wide", page_title="HPC Reasoning Pipeline", page_icon="🧠")

# CSS personnalisé pour un look "IA PRO"
st.markdown("""
<style>
    /* Style général */
    .stApp {
        background-color: #f0f2f6;
    }
    /* Style des conteneurs de réponse */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stTickBar"] {
        background-color: #ffffff;
    }
    .st-emotion-cache-1r6slb0 {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
    }
    .st-emotion-cache-1r6slb0:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        border: 1px solid #cccccc;
    }
    /* Style du bouton principal */
    .stButton > button {
        border-radius: 20px;
        border: 2px solid #ff4b4b;
        color: #ff4b4b;
        background-color: transparent;
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        border-color: #ffffff;
        color: #ffffff;
        background-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialisation du Session State ---
if 'user_question' not in st.session_state:
    st.session_state.user_question = "Comment optimiser l'utilisation des registres vectoriels AVX-512 pour un noyau de calcul stencil 3D avec des conditions aux limites irrégulières tout en minimisant les bank conflicts sur une architecture Intel Sapphire Rapids, et quelle serait la métrique de performance critique à surveiller dans VTune pour ce cas précis ?"
if 'log_output' not in st.session_state:
    st.session_state.log_output = ""
if 'responses' not in st.session_state:
    st.session_state.responses = {}

# --- Barre Latérale (SIDEBAR) ---
with st.sidebar:
    st.image("https://www.via-domitia.fr/sites/default/files/styles/logo/public/atoms/logo/upvd_rvb.png?itok=yq5g8mgs", width=150)
    st.title("Configuration du Pipeline")
    st.divider()
    
    ontology_path = st.text_input("Chemin vers l'ontologie (.ttl):", "ontology/hpc.ttl")

    @st.cache_data
    def load_graph(path):
        try:
            g = Graph().parse(path, format="turtle")
            return g
        except Exception:
            return None
    
    graph = load_graph(ontology_path)
    if graph:
        st.success(f"Ontologie chargée ({len(graph)} faits)")
    else:
        st.error("Ontologie invalide ou introuvable.")
    
    st.divider()
    st.header("Cas d'Étude")

    benchmark_questions = {
        "Choisissez un benchmark...": st.session_state.user_question,
        "1. Arrondis Flottants": "Comment configurer l'arrondi IEEE 754 pour minimiser l'erreur accumulée dans une somme de Kahan sur des données CFD, tout en respectant la norme OpenMP 5.2 ?",
        "2. Alignement Hétérogène": "Quel padding appliquer à un struct {double x; int y;} pour éviter le false-sharing entre threads OpenMP sur un AMD EPYC 9654 ?",
        "3. Communication Hybride": "Comment optimiser MPI_Send/MPI_Recv entre 1 processus MPI par socket (2x EPYC 9654) avec 8 threads OpenMP chacun, en utilisant les shared-memory windows et les hints MPI_Info ?",
        "4. Vectorisation Critique": "Quel pragma OpenMP SIMD utiliser pour vectoriser une boucle avec dépendance 'simd reduction(+:sum)' sur compilateur GCC 13, et comment vérifier l'utilisation des registres ZMM d'AVX-512 ?",
        "5. Synchronisation NUMA-Aware": "Quelle combinaison de omp_lock_t/omp_nest_lock_t choisir pour protéger une hashtable distribuée sur 4 NUMA nodes, avec un pattern d'accès 80% lecture ?"
    }
    
    def on_question_select():
        selected_title = st.session_state.benchmark_selector
        st.session_state.user_question = benchmark_questions[selected_title]
    
    st.selectbox(
        "Sélectionnez un cas d'étude pour la présentation :",
        options=list(benchmark_questions.keys()),
        key="benchmark_selector",
        on_change=on_question_select,
    )
    st.divider()

# --- INTERFACE PRINCIPALE ---
st.title("🧠 Pipeline d'Analyse et de Raisonnement pour le HPC")
st.markdown("Une démonstration de l'augmentation des LLMs par des graphes de connaissances pour des problèmes d'expertise.")

user_question = st.text_area("**Posez votre question experte ici :**", value=st.session_state.user_question, height=175, key="main_question_area")

if st.button("Lancer l'analyse comparative", type="primary", use_container_width=True):
    if user_question.strip() and graph:
        with st.spinner("Analyse en cours... Les experts comparent leurs approches."):
            log_stream.truncate(0); log_stream.seek(0)
            pipeline_logger.info(f"--- NOUVELLE REQUÊTE: '{user_question}' ---")
            
            # Lancement des 3 modes
            st.session_state.responses['direct'] = get_llm_direct_response(user_question)
            
            facts_mode2 = retrieve_relevant_facts(user_question, graph)
            st.session_state.responses['enriched'] = generate_enriched_prompt_response(user_question, facts_mode2)
            st.session_state.responses['enriched_facts'] = facts_mode2
            
            reasoning_report = find_reasoning_path(user_question, graph)
            st.session_state.responses['reasoning'] = generate_response_from_reasoning_path(reasoning_report)
            st.session_state.responses['reasoning_facts'] = reasoning_report
            
            st.session_state.log_output = log_stream.getvalue()
    else:
        st.warning("Veuillez poser une question et charger une ontologie valide.")

# Affichage des résultats en 3 colonnes
if st.session_state.responses.get('direct'):
    st.header("Analyse Comparative des Réponses")
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("1️⃣ LLM Seul (Baseline)")
            st.markdown(st.session_state.responses['direct'])

    with col2:
        with st.container(border=True):
            st.subheader("2️⃣ LLM + Faits (Mots-clés)")
            st.markdown(st.session_state.responses['enriched'])
            with st.expander("Faits extraits par cette méthode"):
                st.write(st.session_state.responses['enriched_facts'] or "Aucun fait trouvé.")

    with col3:
        with st.container(border=True):
            st.subheader("🧠 Notre Expert Augmenté")
            st.markdown(st.session_state.responses['reasoning'])
            with st.expander("Détails du raisonnement"):
                st.json(st.session_state.responses['reasoning_facts'] or {})

# Affichage des logs
if st.session_state.log_output:
    st.divider()
    with st.expander("📋 Afficher les Logs de Traçabilité"):
        st.code(st.session_state.log_output)