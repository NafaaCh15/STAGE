import os
import requests
import logging
import json
from typing import Optional

# Use a specific logger for this module
logger = logging.getLogger("pipeline_trace." + __name__)

# Constants
DEFAULT_HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_API_BASE_URL = "https://api-inference.huggingface.co/models/"

# Default error message for user-facing scenarios if needed elsewhere,
# though this function primarily returns None on error for programmatic handling.
HF_CALLER_DEFAULT_ERROR_MSG = "Erreur de communication avec le service Hugging Face Inference API."


def call_hf_inference_api(prompt: str, model_name: str = DEFAULT_HF_MODEL) -> Optional[str]:
    """
    Calls the Hugging Face Inference API with a given prompt and model.

    Args:
        prompt (str): The input prompt to send to the LLM.
        model_name (str, optional): The name of the Hugging Face model to use.
                                    Defaults to DEFAULT_HF_MODEL.

    Returns:
        Optional[str]: The generated text from the LLM if successful, otherwise None.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set. Cannot call Hugging Face Inference API.")
        return None

    api_url = f"{HF_API_BASE_URL}{model_name}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    # Parameters can be adjusted based on model and desired output.

    # Formatter for Llama-2-chat style prompts
    # The user's raw prompt is passed as {prompt}
    formatted_prompt = (
        f"<s>[INST] <<SYS>>\n"
        f"Tu es un assistant expert en HPC qui répond de manière claire et concise en se basant sur les faits fournis.\n"
        f"<</SYS>>\n\n"
        f"{prompt} [/INST]"
    )

    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "return_full_text": False,
            "max_new_tokens": 1024,
            "temperature": 0.5,
            # "top_p": 0.9,        # Optional, for sampling
            # "do_sample": True    # Optional, for sampling
        },
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }

    logger.info(f"Calling Hugging Face Inference API for model: {model_name}")
    # Log only part of the *original user* prompt to avoid extremely long log messages that include the full formatted prompt.
    # The full formatted prompt is implicitly tested by successful calls.
    prompt_summary = prompt[:150] + "..." if len(prompt) > 150 else prompt
    logger.debug(f"API URL: {api_url}, Original User Prompt (summary): {prompt_summary}")
    # For debugging the exact format sent:
    # logger.debug(f"Full Formatted Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=45)
        response.raise_for_status()

        try:
            response_data = response.json()
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON response from API. Status: {response.status_code}, Body: {response.text}")
            return None

        logger.debug(f"Raw API response data: {response_data}")

        if isinstance(response_data, list) and response_data:
            if isinstance(response_data[0], dict) and "generated_text" in response_data[0]:
                generated_text = response_data[0]["generated_text"]
                logger.info("Successfully received and parsed 'generated_text' from Hugging Face API.")
                return generated_text.strip()
            else:
                logger.warning(f"Unexpected item structure in response list: {response_data[0]}. Looking for 'generated_text'.")
        elif isinstance(response_data, dict) and "generated_text" in response_data :
            generated_text = response_data["generated_text"]
            logger.info("Successfully received and parsed 'generated_text' (direct dict) from Hugging Face API.")
            return generated_text.strip()
        elif isinstance(response_data, dict) and "error" in response_data:
            error_message = response_data.get("error")
            estimated_time = response_data.get("estimated_time")
            if estimated_time:
                 logger.warning(f"Model {model_name} is currently loading. Estimated time: {estimated_time}s. Error: {error_message}")
                 return None
            logger.error(f"API returned JSON with error: {error_message}")
            return None

        logger.error(f"Unexpected JSON response structure from API: {response_data}")
        return None

    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        error_text = http_err.response.text
        logger.error(f"HTTP error occurred: {http_err} - Status: {status_code}")
        try:
            error_details = http_err.response.json()
            logger.error(f"Error details from API (JSON): {error_details}")
            if status_code == 503 and 'estimated_time' in error_details:
                 estimated_time = error_details.get('estimated_time')
                 logger.warning(f"Model {model_name} is loading. Estimated time: {estimated_time}s. Consider retrying.")
            elif 'error' in error_details:
                 logger.error(f"Specific error from API: {error_details['error']}")
        except json.JSONDecodeError:
            logger.error(f"Error details from API (non-JSON): {error_text}")

        if status_code == 401:
            logger.error("Authentication error (401): Invalid HF_TOKEN or token does not have permissions for this model.")
        elif status_code == 429:
            logger.error("Rate limit error (429): Too many requests. Please wait and try again.")
        elif status_code == 503:
            logger.warning(f"Service unavailable (503) for model {model_name}. The model might be loading or temporarily down.")
        return None

    except requests.exceptions.Timeout:
        logger.error(f"Request timed out while calling Hugging Face API for model {model_name}.")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"A request error occurred: {req_err}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred in call_hf_inference_api: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    # Configure the pipeline_trace logger for direct script testing
    test_logger_hf = logging.getLogger("pipeline_trace") # Get the parent logger
    if not test_logger_hf.handlers: # Add console handler if no handlers are configured
        test_logger_hf.setLevel(logging.DEBUG) # Set to DEBUG to see all messages from this module
        ch = logging.StreamHandler()
        ch_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(ch_formatter)
        test_logger_hf.addHandler(ch)
        test_logger_hf.propagate = False

    logger.info("Running hf_llm_caller.py directly for testing.") # Use module-level logger

    hf_token_available = os.getenv("HF_TOKEN")
    if not hf_token_available:
        print("\n[IMPORTANT] HF_TOKEN environment variable is not set.")
        print("Please set HF_TOKEN to your Hugging Face API token to run this test.")
        print("You can get a token from Hugging Face settings: https://huggingface.co/settings/tokens")
        print("This test will attempt to call a public model, which requires a token for API access.")
    else:
        logger.info(f"HF_TOKEN is set. Proceeding with a sample API call using model: {DEFAULT_HF_MODEL}")

        test_prompt_1 = "Translate 'hello world' to French."
        logger.info(f"\n--- Test Case 1: Simple Translation ---")
        logger.info(f"Prompt: \"{test_prompt_1}\"")
        response_1 = call_hf_inference_api(test_prompt_1)
        if response_1:
            logger.info(f"LLM Response: \"{response_1}\"")
        else:
            logger.error("Failed to get a response or an error occurred for Test Case 1.")

        test_prompt_2 = "What is the capital of France?"
        logger.info(f"\n--- Test Case 2: Factual Question ---")
        logger.info(f"Prompt: \"{test_prompt_2}\"")
        response_2 = call_hf_inference_api(test_prompt_2, model_name=DEFAULT_HF_MODEL)
        if response_2:
            logger.info(f"LLM Response: \"{response_2}\"")
        else:
            logger.error("Failed to get a response or an error occurred for Test Case 2.")

        logger.info(f"\n--- Test Case 3: Non-existent model ---")
        non_existent_model = "this/model-does-not-exist-12345"
        logger.info(f"Prompt: \"Hello\" (Model: {non_existent_model})")
        response_3 = call_hf_inference_api("Hello", model_name=non_existent_model)
        if response_3 is None:
            logger.info("Correctly handled non-existent model (returned None).")
        else:
            logger.error(f"Unexpectedly got a response for non-existent model: {response_3}")

    logger.info("Finished testing hf_llm_caller.py directly.")
