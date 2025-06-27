import logging
from typing import Dict

from .hf_llm_caller import call_hf_inference_api, DEFAULT_HF_MODEL

logger = logging.getLogger("pipeline_trace." + __name__)

DEFAULT_ERROR_RESPONSE = "Je suis désolé, je ne peux pas traiter cette demande pour le moment en raison d'un problème technique."

def generate_response_from_reasoning_path(reasoning_result: Dict) -> str:
    """
    Génère une réponse experte et fluide en se basant sur les faits du graphe.
    """
    question = reasoning_result.get("question")
    deduced_facts = reasoning_result.get("faits_deduits", [])

    if not deduced_facts:
        return "Le raisonneur n'a trouvé aucun fait pertinent dans l'ontologie pour répondre."

    facts_str_for_prompt = "\n".join([f"- {fact}" for fact in deduced_facts])

    # --- PROMPT EXPERT FINAL V5 ---
    prompt = (
    f"Tu es un architecte HPC utilisant un graphe de connaissances certifié. Réponds EXCLUSIVEMENT en exploitant :\n"
    f"- Les relations sémantiques du graphe (ex: :FalseSharing rdf:type :ProblemePerformance)\n"  
    f"- Les propriétés techniques (ex: :aTailleCacheLine, :aCompatibiliteCompilateurs)\n"
    f"- Les exemples de code liés (via :aPourExemple)\n\n"
    f"### Question :\n\"{question}\"\n\n"
    f"### Contexte Structuré :\n{facts_str_for_prompt}\n\n"
    f"### Règles Stricts :\n"
    f"1. **Exploitation des Relations** : Mentionne explicitement comment les concepts sont connectés dans l'ontologie.\n"
    f"    Ex: 'Le graphe lie :FalseSharing à :CacheLineAlignment via :estUneSolutionPour.'\n"
    f"2. **Granularité Numérique** : Cite toujours les valeurs exactes (tailles de cache, padding, etc.).\n"
    f"3. **Code et Commandes** : Intègre les snippets avec leur contexte ontologique.\n"
    f"    Ex: 'Comme montré dans :ExempleCodePaddingStruct (lié à :PaddingDeStruct), utilisez...'\n"
    f"4. **Validation Croisée** : Corrèle chaque assertion avec un fait du graphe.\n\n"
    f"**Style Requis** :\n"
    f"- Texte continu mais avec segments courts et techniques.\n"
    f"- Termes clés en *italique* pour les concepts de l'ontologie.\n"
    f"- Citations explicites comme '[Graphe: :AMD_EPYC_9654 :aTailleCacheLine \"64 octets\"]'.\n\n"
    f"Réponse :"
)
    logger.info(f"Prompt Expert Final V5 (longueur: {len(prompt)})...")
    
    generated_text = call_hf_inference_api(prompt, model_name=DEFAULT_HF_MODEL)
    
    if generated_text is None:
        return DEFAULT_ERROR_RESPONSE
    
    return clean_llm_nl_response(generated_text)

def clean_llm_nl_response(llm_output: str) -> str:
    """Nettoie la sortie brute en langage naturel d'un LLM."""
    if not llm_output:
        return ""
    cleaned_response = llm_output.strip()
    if cleaned_response.lower().startswith("réponse d'expert:"):
        cleaned_response = cleaned_response[len("réponse d'expert:"):].strip()
    return cleaned_response

def get_llm_direct_response(user_question: str, model_name: str = DEFAULT_HF_MODEL) -> str:
    """Génère une réponse directe d'un LLM sans contexte d'ontologie."""
    prompt = (f"Répondez directement et concisement à la question suivante en français, en utilisant vos connaissances générales :\n\"{user_question}\"")
    generated_text = call_hf_inference_api(prompt, model_name=model_name)
    if generated_text is None:
        return DEFAULT_ERROR_RESPONSE
    return clean_llm_nl_response(generated_text)