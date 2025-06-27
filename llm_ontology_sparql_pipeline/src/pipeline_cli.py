import logging
import os

import logging
import os
from typing import List, Optional # Added Optional for type hinting

# Import necessary functions from other modules within the 'src' package.
from .llm_sparql_generator import generate_sparql_query
from .sparql_executor import execute_sparql_query
from .llm_response_generator import generate_natural_language_response, DEFAULT_ERROR_RESPONSE as NL_DEFAULT_ERROR_RESPONSE
from .ontology_parser import extract_schema_from_ttl # Import the new schema extractor

# Configure basic logging for the CLI application.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)

# --- Configuration Constants ---
# TODO: Consider moving these to a dedicated configuration file (e.g., config.py or .env).

# Maximum attempts for SPARQL query generation if the first attempt yields no results.
MAX_SPARQL_ATTEMPTS = 2

# ONTOLOGY_PATH:
# Defines the location of the ontology file.
ONTOLOGY_DIR = os.path.join(os.path.dirname(__file__), '..', 'ontology')
ONTOLOGY_FILE_NAME = "medical.ttl"
ONTOLOGY_PATH = os.path.join(ONTOLOGY_DIR, ONTOLOGY_FILE_NAME)


def main_pipeline():
    """
    Orchestrates the main LLM-Ontology-SPARQL pipeline flow:
    1. Checks for OpenAI API key.
    2. Verifies ontology file existence and extracts its schema.
    3. Takes a user's question.
    4. Attempts to generate a SPARQL query using the extracted schema (with retries).
    5. Executes the SPARQL query.
    6. Generates a natural language response based on query results.
    7. Prints intermediate and final outputs.
    """
    logging.info("Starting the LLM Ontology SPARQL Pipeline CLI...")

    # Critical check: OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("CRITICAL: OPENAI_API_KEY environment variable is not set.")
        print("Erreur: La clé d'API OpenAI (OPENAI_API_KEY) n'est pas configurée. Veuillez la définir pour utiliser le pipeline.")
        return

    # Verify ontology file existence and load its schema
    if not os.path.exists(ONTOLOGY_PATH):
        logging.error(f"CRITICAL: Ontology file not found at: {ONTOLOGY_PATH}")
        print(f"Erreur: Le fichier d'ontologie '{ONTOLOGY_PATH}' est introuvable. Assurez-vous qu'il existe.")
        return
    logging.info(f"Using ontology file: {ONTOLOGY_PATH}")

    ontology_schema = extract_schema_from_ttl(ONTOLOGY_PATH)
    if ontology_schema is None: # Handles errors from schema extraction (e.g. parsing error)
        logging.error(f"CRITICAL: Failed to extract schema from ontology file {ONTOLOGY_PATH}.")
        print(f"Erreur: Impossible d'extraire le schéma du fichier d'ontologie '{ONTOLOGY_PATH}'. Vérifiez le fichier et les logs.")
        return
    if not ontology_schema.strip(): # Handles case where schema is empty string
        logging.warning(f"Extracted ontology schema from {ONTOLOGY_PATH} is empty. LLM might struggle to generate queries.")
        # Proceed, but LLM performance might be degraded. Could also choose to exit.
        print("Avertissement: Le schéma extrait de l'ontologie est vide. La génération de requêtes SPARQL risque d'être de mauvaise qualité.")

    logging.info(f"Successfully extracted ontology schema:\n{ontology_schema}")


    # Get user input
    user_question = input("Posez votre question en français (ex: Quels médicaments traitent la COVID19 ?) : ")
    if not user_question.strip():
        logging.warning("No question provided. Exiting.")
        print("Aucune question n'a été posée. Arrêt du programme.")
        return
    logging.info(f"User question received: '{user_question}'")

    sparql_query: Optional[str] = None
    sparql_results: List[str] = []
    current_prompt_context = "" # For adding context on retry

    for attempt in range(1, MAX_SPARQL_ATTEMPTS + 1):
        logging.info(f"--- SPARQL Generation Attempt {attempt}/{MAX_SPARQL_ATTEMPTS} ---")

        current_schema_for_llm = ontology_schema # Use dynamically loaded schema
        if attempt > 1 and current_prompt_context: # Add context only if available and it's a retry
            # This is a very basic way to add context. More sophisticated methods could be used.
            # Example: Prepend context to the schema or append to user question.
            # For now, let's assume the LLM can use this additional context if provided within the schema string
            # or as part of an augmented user question.
            # A simple way: just pass a modified schema or question.
            # Here, we'll augment the schema string for simplicity.
            augmented_schema = f"{ontology_schema}\n\nAdditional context for retry:\n{current_prompt_context}"
            logging.info("Retrying SPARQL generation with additional context in prompt.")
            generated_query_attempt = generate_sparql_query(user_question, augmented_schema)
        else:
            generated_query_attempt = generate_sparql_query(user_question, ontology_schema)


        if not generated_query_attempt:
            logging.error(f"SPARQL query generation failed on attempt {attempt} (LLM returned None or error).")
            if attempt == MAX_SPARQL_ATTEMPTS:
                print("Désolé, je n'ai pas pu générer de requête SPARQL pour votre question après plusieurs tentatives.")
                final_response = generate_natural_language_response(user_question, []) # Try to give a final answer
                print(f"\nRéponse finale :\n{final_response if final_response else NL_DEFAULT_ERROR_RESPONSE}")
                return
            current_prompt_context = "The previous attempt to generate a SPARQL query failed or returned an invalid query. Please ensure the query is valid and directly answers the user's question using only the provided ontology schema."
            continue

        sparql_query = generated_query_attempt
        print(f"\n[DEBUG Attempt {attempt}] Generated SPARQL Query:\n{sparql_query}\n")

        try:
            sparql_results = execute_sparql_query(sparql_query, ONTOLOGY_PATH)
            logging.info(f"SPARQL query executed on attempt {attempt}. Number of results: {len(sparql_results)}.")
            print(f"[DEBUG Attempt {attempt}] SPARQL Execution Results: {sparql_results}\n")
        except Exception as e:
            logging.error(f"Exception during SPARQL execution on attempt {attempt}: {e}", exc_info=True)
            # It's a critical error if SPARQL execution fails with a valid-looking query.
            print("Désolé, une erreur est survenue lors de l'exécution de la requête SPARQL.")
            final_response = generate_natural_language_response(user_question, [])
            print(f"\nRéponse finale :\n{final_response if final_response else NL_DEFAULT_ERROR_RESPONSE}")
            return

        is_select_query = sparql_query.strip().upper().startswith("SELECT")
        if is_select_query and not sparql_results and attempt < MAX_SPARQL_ATTEMPTS:
            logging.warning(f"Attempt {attempt}: SELECT query yielded no results. Will retry SPARQL generation.")
            current_prompt_context = "The previously generated SPARQL query (a SELECT query) returned no results. Please try to formulate a different SPARQL query that might yield results for the user's question, based on the provided ontology schema. Consider alternative properties, class relationships, or ensure correct entity URIs/literals are used."
        else:
            if is_select_query and not sparql_results:
                 logging.warning(f"Attempt {attempt}: SELECT query yielded no results, and it's the final attempt.")
            break

    if not sparql_query: # Should be caught by earlier checks, but as a safeguard
        logging.error("Failed to obtain a SPARQL query after all attempts.")
        final_response = generate_natural_language_response(user_question, [])
        print(f"\nRéponse finale :\n{final_response if final_response else NL_DEFAULT_ERROR_RESPONSE}")
        return

    logging.info("Stage 3: Generating natural language response...")
    final_response = generate_natural_language_response(user_question, sparql_results)

    if not final_response or final_response == NL_DEFAULT_ERROR_RESPONSE :
        logging.warning("Natural language response generation returned a default error or None.")
        # Depending on desired strictness, we might just print the default error or the SPARQL results.
        print(f"\nRéponse finale (limitée) :\n{NL_DEFAULT_ERROR_RESPONSE}")
        if sparql_results: # If we have SPARQL results, maybe show them if NL fails
             print(f"Voici les données brutes trouvées : {sparql_results}")
    else:
        print(f"\nRéponse finale :\n{final_response}")

    logging.info("Pipeline execution completed.")

if __name__ == "__main__":
    print("--- LLM Ontology SPARQL Pipeline CLI ---")
    print("NOTE: This script may make multiple calls to the OpenAI API, which can incur costs.")
    print("Ensure your OPENAI_API_KEY environment variable is set correctly.")
    main_pipeline()
