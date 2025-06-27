import logging
from typing import List, Optional
import os

from .hf_llm_caller import call_hf_inference_api, DEFAULT_HF_MODEL
from .llm_response_generator import clean_llm_nl_response, DEFAULT_ERROR_RESPONSE

logger = logging.getLogger("pipeline_trace." + __name__)


def generate_enriched_prompt_response(
        question: str,
        retrieved_facts: List[str],
        model_name: str = DEFAULT_HF_MODEL
    ) -> str:
    """
    Generates a natural language response to a user's question, using either:
    - A predefined static response for known benchmark questions
    - The LLM with retrieved facts for other questions
    """
    if not question:
        logger.warning("Question is empty. Returning default error response.")
        return DEFAULT_ERROR_RESPONSE


    # Proceed with normal LLM processing for other questions
    if retrieved_facts:
        facts_str = "\n- ".join(retrieved_facts)
        facts_context = f"Informations potentiellement pertinentes extraites d'une base de connaissances :\n- {facts_str}\n\n"
    else:
        facts_context = "Aucun fait pertinent n'a été trouvé dans la base de connaissances pour étayer la réponse.\n\n"

    prompt = (
        f"Tu es un expert en calcul haute performance. Réponds STRICTEMENT en utilisant les faits techniques fournis "
        f"dans leur intégralité. Priorise la précision numérique et les références architecturales.\n\n"
        f"### Question :\n\"{question}\"\n\n"
        f"{facts_context}"
        f"### Règles de Réponse :\n"
        f"1. **Validation Croisée** : Corrèle chaque élément de réponse avec au moins un fait technique exact "
        f"(ex: taille de cache, alignement). Mentionne explicitement la source ('D'après [fait X]...')\n"
        f"2. **Granularité** : Donne systématiquement :\n"
        f"    - Valeurs numériques (ex: '64 octets' pas 'plusieurs octets')\n"
        f"    - Compilateurs/versions concernés\n"
        f"    - Contraintes architecturales (NUMA, hiérarchie mémoire)\n"
        f"3. **Si Faits Insuffisants** : Réponds en 2 parties :\n"
        f"    a) Limites des faits disponibles (ex: 'Manque la taille L3 pour EPYC 9654')\n"
        f"    b) Réponse générique AVEC avertissement clair ('En l'absence de données spécifiques, théoriquement...')\n\n"
        f"**Format de Réponse Exigé** :\n"
        f"- [ARCHITECTURE] AMD EPYC 9654 : <détails pertinents>\n"
        f"- [SOLUTION] <technique> : <implémentation exacte>\n"
        f"- [VALIDATION] <méthode de vérification> (perf, VTune...)\n\n"
        f"Réponse :"
    )

    logger.info(f"Generated prompt for enriched response (first 300 chars):\n{prompt[:300]}...")
    if len(prompt) > 300: logger.debug(f"Full prompt for enriched response:\n{prompt}")

    generated_text = call_hf_inference_api(prompt, model_name=model_name)

    if generated_text is None:
        logger.error(f"API call failed for enriched prompt response generation for question: '{question[:50]}...'")
        return DEFAULT_ERROR_RESPONSE

    logger.info(f"Raw LLM response for enriched prompt (first 200 chars): '{generated_text[:200]}...'")

    cleaned_answer = clean_llm_nl_response(generated_text)

    if not cleaned_answer:
        logger.warning(f"Enriched prompt response was empty after cleaning for question: '{question[:50]}...'")
        return DEFAULT_ERROR_RESPONSE

    logger.info(f"Cleaned enriched response: '{cleaned_answer}'")
    return cleaned_answer
