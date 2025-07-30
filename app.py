import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Système Multi-Agents avec Gemini",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styles CSS pour améliorer le design (Thème Sombre et Responsive) ---
st.markdown("""
<style>
    /* Thème général sombre */
    .stApp {
        background-color: #0F172A; /* Fond bleu nuit */
        color: #CBD5E1; /* Texte gris clair par défaut */
    }

    /* Titres */
    h1, h2, h3 {
        color: #FFFFFF; /* Titres en blanc */
    }

    /* Boutons */
    .stButton>button {
        background-color: #3B82F6; /* Bleu vif */
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #2563EB; /* Bleu plus foncé au survol */
    }

    /* Zones de saisie */
    .stTextArea textarea, .stTextInput input {
        background-color: #1E293B;
        color: #E2E8F0;
        border: 1px solid #334155;
        border-radius: 8px;
    }

    /* Expanders pour les résultats des agents */
    .stExpander {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
    }
    .stExpander header {
        color: #F1F5F9;
        font-weight: bold;
    }
    
    /* Amélioration de la visibilité des messages d'état */
    .stAlert {
        border-radius: 8px;
    }

    /* Barre latérale */
    .st-emotion-cache-16txtl3 {
        background-color: #1E293B;
    }

    /* --- Media Queries pour la responsivité --- */
    @media (max-width: 768px) {
        h1 {
            font-size: 2.2rem; /* Réduire la taille du titre sur mobile */
        }
        .stButton>button {
            width: 100%; /* Faire en sorte que le bouton prenne toute la largeur */
            margin-top: 10px;
        }
    }
</style>
""", unsafe_allow_html=True)


# --- Fonctions des Agents ---

def configure_gemini(api_key):
    """Configure l'API Gemini avec la clé fournie."""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Erreur lors de la configuration de l'API Gemini : {e}")
        return False

def get_gemini_response(prompt, agent_name):
    """
    Fonction générique pour appeler le modèle Gemini et gérer les erreurs.
    Affiche un spinner pendant la génération.
    """
    with st.spinner(f"🤖 L'agent **{agent_name}** est en train de travailler..."):
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'appel à l'API Gemini pour l'agent {agent_name}: {e}")
            return None

def planner_agent(task):
    """
    Agent Planner : Décompose la tâche complexe en un plan détaillé.
    """
    prompt = f"""
    Rôle : Vous êtes un planificateur expert en IA. Votre mission est de décomposer une tâche complexe en une série d'étapes claires, logiques et réalisables.
    Ne générez pas le contenu, fournissez uniquement le plan.

    Tâche complexe à planifier : "{task}"

    Format de sortie attendu :
    **Plan d'action :**
    1.  **Étape 1 :** [Description de la première étape]
    2.  **Étape 2 :** [Description de la deuxième étape]
    3.  ...
    N.  **Étape N :** [Description de la dernière étape]

    Générez le plan pour la tâche ci-dessus.
    """
    return get_gemini_response(prompt, "Planner")

def executor_agent(task, plan):
    """
    Agent Executor : Exécute le plan pour accomplir la tâche.
    """
    prompt = f"""
    Rôle : Vous êtes un agent d'exécution IA. Votre mission est de suivre un plan d'action pour accomplir une tâche spécifique et de produire le résultat final.

    Tâche originale : "{task}"
    Plan à exécuter :
    ---
    {plan}
    ---

    Instructions :
    - Suivez chaque étape du plan attentivement.
    - Générez une réponse complète et bien structurée qui répond à la tâche originale.
    - Assurez-vous que le contenu est de haute qualité, précis et pertinent.

    Produisez maintenant le résultat final.
    """
    return get_gemini_response(prompt, "Executor")

def critic_agent(task, plan, execution_result):
    """
    Agent Critic : Évalue la qualité de l'exécution et fournit des retours.
    """
    prompt = f"""
    Rôle : Vous êtes un critique expert en IA. Votre mission est d'évaluer la qualité d'un travail effectué par un autre agent IA (Executor) en se basant sur la tâche originale et le plan fourni.

    Tâche originale : "{task}"
    Plan suivi :
    ---
    {plan}
    ---
    Résultat de l'exécution à évaluer :
    ---
    {execution_result}
    ---

    Instructions d'évaluation :
    1.  **Conformité au plan :** Le résultat suit-il toutes les étapes du plan ?
    2.  **Qualité et Précision :** Le contenu est-il de haute qualité, précis, et sans erreurs factuelles ?
    3.  **Complétude :** La réponse est-elle complète par rapport à la tâche originale ?
    4.  **Clarté et Structure :** Le résultat est-il bien structuré, clair et facile à comprendre ?

    Format de sortie attendu :
    **Évaluation critique :**
    - **Points forts :** [Listez les aspects positifs du résultat]
    - **Points à améliorer :** [Listez les faiblesses, les incohérences ou les omissions]
    - **Suggestion d'amélioration :** [Proposez des modifications concrètes pour améliorer le résultat]
    - **Note globale (sur 10) :** [Donnez une note justifiée]
    """
    return get_gemini_response(prompt, "Critic")


# --- Interface Streamlit ---

st.title("🤖 Système Multi-Agents avec Gemini 1.5 Flash")
st.markdown("Cette application utilise une chaîne d'agents IA (Planner, Executor, Critic) pour traiter des tâches complexes.")

# --- Barre latérale pour la configuration et l'état de l'API ---
api_key = os.getenv("GOOGLE_API_KEY")

with st.sidebar:
    st.header("Configuration")
    if api_key:
        st.success("Clé API chargée depuis le fichier .env !", icon="✅")
    else:
        st.error("Clé API non trouvée.", icon="⚠️")
        st.info("Veuillez créer un fichier `.env` à la racine de votre projet et y ajouter votre clé API comme ceci :\n`GOOGLE_API_KEY='VOTRE_CLE_API'`")

    st.markdown("---")
    st.markdown("### Comment ça marche ?")
    st.markdown("""
    1.  **Planner** : Décompose votre tâche en un plan.
    2.  **Executor** : Réalise le plan et génère la réponse.
    3.  **Critic** : Évalue la réponse pour en assurer la qualité.
    """)

# --- Zone de contenu principale ---
st.header("1. Soumettez votre tâche complexe")
task_input = st.text_area("Décrivez la tâche que vous souhaitez que les agents accomplissent :", height=150,
                          placeholder="Exemple : Rédige un article de blog sur l'impact de l'IA sur le marché du travail, en incluant les opportunités et les défis.")

if st.button("Lancer les agents", type="primary"):
    if not api_key:
        st.error("Veuillez configurer votre clé API dans le fichier .env pour continuer.")
    elif not task_input:
        st.warning("Veuillez entrer une tâche à accomplir.")
    else:
        # Configuration de l'API
        if configure_gemini(api_key):
            st.success("API Gemini configurée avec succès !")
            st.markdown("---")
            st.header("2. Analyse et Traitement par les Agents")

            # --- Étape 1: Planner ---
            plan = planner_agent(task_input)
            if plan:
                with st.expander("📝 **Plan généré par l'Agent Planner**", expanded=True):
                    st.markdown(plan)

                # --- Étape 2: Executor ---
                execution_result = executor_agent(task_input, plan)
                if execution_result:
                    st.markdown("---")
                    st.subheader("✅ Résultat produit par l'Agent Executor")
                    st.markdown(execution_result)

                    # --- Étape 3: Critic ---
                    critic_review = critic_agent(task_input, plan, execution_result)
                    if critic_review:
                        with st.expander("🧐 **Évaluation de l'Agent Critic**", expanded=True):
                            st.markdown(critic_review)
