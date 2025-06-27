import logging
import re
from typing import List, Set, Tuple, Optional, Dict

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, RDF, XSD # XSD might be useful for formatting literals

logger = logging.getLogger("pipeline_trace." + __name__)

# Simple stopword lists (can be expanded)
STOPWORDS_FR = set([
    "le", "la", "les", "de", "des", "du", "et", "ou", "est", "sont", "un", "une",
    "en", "pour", "que", "qui", "quoi", "quel", "quelle", "quels", "quelles",
    "avec", "sans", "dans", "sur", "sous", "par", "ce", "cet", "cette", "ces",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "nos",
    "votre", "vos", "leur", "leurs", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "ai", "as", "a", "avons", "avez", "ont", "suis", "es", "sommes", "etes",
    "étais", "était", "étions", "étiez", "étaient", "serai", "seras", "sera",
    "serons", "serez", "seront", "comment", "pourquoi", "quand", "où", "y", "a-t-il"
])
STOPWORDS_EN = set([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "can", "could", "may", "might", "must",
    "and", "or", "but", "if", "of", "at", "by", "for", "with", "about", "to", "in", "on",
    "what", "which", "who", "whom", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "mine", "yours", "hers", "ours", "theirs", "how", "why", "when", "where"
])
# Combine for broader filtering, or detect language if more sophisticated
COMBINED_STOPWORDS = STOPWORDS_FR.union(STOPWORDS_EN)


def _extract_keywords(question: str, stopwords: Set[str] = COMBINED_STOPWORDS) -> Set[str]:
    """
    Extracts a set of keywords from a question string.
    Basic implementation: lowercase, remove punctuation, remove stopwords, take unique words.
    """
    if not question:
        return set()

    # Preserve hyphens in words, remove other punctuation, and lowercase
    # This regex keeps alphanumeric characters, spaces, and hyphens that are part of words.
    processed_question = re.sub(r'[^\w\s-]|(?<!\w)-(?!\w)|-(?<!\w)', '', question.lower())
    # A simpler regex that keeps all hyphens: re.sub(r'[^\w\s-]', '', question.lower())
    # Let's use the simpler one for now, as the more complex one might be too restrictive.
    processed_question = re.sub(r'[^\w\s-]', '', question.lower())


    words = processed_question.split()
    keywords = {word for word in words if word not in stopwords and len(word) > 2}

    # Handle cases like "covid-19" -> add "covid" and "19" as well, or "covid-19" itself.
    # The current regex `[^\w\s-]` preserves "covid-19" as a single token if not split by space.
    # If "covid-19" is a keyword, we also want to match "covid19" if that's how it's in the data, or vice-versa.
    # For now, the keyword search is `keyword in literal_value_lower`.
    # If keyword is "covid-19", it will match "covid-19".
    # If literal is "covid19" and keyword is "covid-19", it won't match.
    # Let's add a variation: if a keyword contains a hyphen, also add its non-hyphenated version.
    additional_keywords = set()
    for kw in keywords:
        if "-" in kw:
            additional_keywords.add(kw.replace("-", ""))
            additional_keywords.add(kw.replace("-", " ")) # also try space separated if model wrote it like that
    keywords.update(additional_keywords)

    logger.debug(f"Extracted keywords from '{question}': {keywords}")
    return keywords

def get_node_label(graph: Graph, node: URIRef, prefixes: Optional[Dict[str, str]] = None) -> str:
    """
    Tries to get a human-readable label for a URI, falling back to qname or URI string.
    """
    if not isinstance(node, URIRef):
        return str(node) # Should not happen if called with URIRef

    # Try common labeling properties
    for label_prop in [RDFS.label, URIRef("http://example.org/medical#nom"), URIRef("http://purl.org/dc/terms/title"), URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")]:
        label = graph.value(subject=node, predicate=label_prop)
        if label:
            return str(label)

    # Fallback to qname or URI
    if prefixes:
        try:
            qname = graph.compute_qname(node, generate=False) # rdflib 6+
            if qname[0] and qname[1]: # Check if prefix and local name exist
                 return f"{qname[0]}:{qname[1]}"
        except Exception: # compute_qname might fail or not find a prefix
            pass
        # Fallback for rdflib < 6 or if compute_qname fails to find a simple qname
        for p, ns in prefixes.items():
            if str(node).startswith(str(ns)):
                return f"{p}:{str(node)[len(str(ns)):]}"

    return f"<{str(node)}>" # Default to full URI

def retrieve_relevant_facts(
        question: str,
        graph: Graph,
        ontology_schema_info: Optional[Dict] = None, # Placeholder for future use
        max_facts: int = 7, # Increased default slightly
        max_subjects_to_process: int = 5
    ) -> List[str]:
    """
    Retrieves relevant facts/triples from the ontology graph based on keywords in the question.

    Args:
        question (str): The user's natural language question.
        graph (Graph): The RDFlib graph object containing the ontology.
        ontology_schema_info (Optional[Dict]): Optional dictionary with schema details
                                              (e.g., from extract_schema_from_ttl) to guide fact extraction.
                                              Currently not deeply integrated.
        max_facts (int): Maximum number of fact strings to return.
        max_subjects_to_process (int): Maximum number of matched subjects to expand facts from.

    Returns:
        List[str]: A list of strings, where each string is a formatted fact relevant to the question.
    """
    logger.info(f"Retrieving relevant facts for question: '{question[:100]}...'")
    keywords = _extract_keywords(question)
    if not keywords:
        logger.info("No keywords extracted from question, returning empty fact list.")
        return []

    # Define common literal properties to search for keywords in.
    # Consider making this configurable or deriving from schema_info if available.
    # For medical.ttl, :nom is important.
    target_literal_properties = {
        RDFS.label,
        RDFS.comment,
        URIRef("http://example.org/medical#nom"), # From medical.ttl
        URIRef("http://purl.org/dc/terms/description"),
        URIRef("http://www.w3.org/2004/02/skos/core#definition")
    }
    if ontology_schema_info and "properties" in ontology_schema_info: # Future use
        # Could add properties known to have literal ranges from schema_info
        pass

    found_subjects: Set[URIRef] = set()
    logger.debug(f"Searching for keywords {keywords} in properties: {target_literal_properties}")

    for s, p, o in graph:
        if p in target_literal_properties and isinstance(o, Literal):
            literal_value_lower = str(o).lower()
            for keyword in keywords:
                if keyword in literal_value_lower: # Simple substring match
                    if isinstance(s, URIRef): # Ensure subject is a URI
                        found_subjects.add(s)
                        logger.debug(f"Keyword '{keyword}' matched literal for subject {s}")
                        break # Move to next triple once a keyword matches this literal

    retrieved_facts: List[str] = []

    # Get prefixes for nicer output of URIs
    prefixes = {prefix: str(ns) for prefix, ns in graph.namespaces() if prefix}

    subjects_processed_count = 0
    for subject_uri in list(found_subjects): # Convert set to list to allow slicing if needed
        if subjects_processed_count >= max_subjects_to_process:
            logger.info(f"Reached max subjects to process ({max_subjects_to_process}). Stopping fact expansion.")
            break

        subject_label = get_node_label(graph, subject_uri, prefixes)

        # Try to get the type of the subject for more context
        subject_types = [get_node_label(graph, type_uri, prefixes) for type_uri in graph.objects(subject_uri, RDF.type) if isinstance(type_uri, URIRef)]
        type_info_str = f" (type: {', '.join(subject_types)})" if subject_types else ""

        # Add facts about this subject
        # Prioritize literals and well-known annotation properties
        priority_props = [RDFS.label, URIRef("http://example.org/medical#nom"), RDFS.comment]

        for p_prop in priority_props:
            for obj_literal in graph.objects(subject_uri, p_prop):
                if isinstance(obj_literal, Literal):
                    prop_label = get_node_label(graph, p_prop, prefixes)
                    fact_str = f"{subject_label}{type_info_str} - {prop_label}: {str(obj_literal)}"
                    if fact_str not in retrieved_facts: # Avoid duplicate facts
                        retrieved_facts.append(fact_str)
                        logger.debug(f"Added priority fact: {fact_str}")
                    if len(retrieved_facts) >= max_facts: break
            if len(retrieved_facts) >= max_facts: break
        if len(retrieved_facts) >= max_facts: continue # Move to next subject if max_facts reached

        # Then add other outgoing properties (object properties or other datatype properties)
        # Limit the number of these "other" properties per subject to avoid too much noise
        other_props_count = 0
        max_other_props_per_subject = 3 # Configurable: how many other relations to show per subject

        for _, p, o in graph.triples((subject_uri, None, None)):
            if p in priority_props: continue # Already handled

            prop_label = get_node_label(graph, p, prefixes)
            obj_str = ""
            if isinstance(o, Literal):
                obj_str = str(o)
                # Could add (datatype: xsd:type) if needed
            elif isinstance(o, URIRef):
                obj_str = get_node_label(graph, o, prefixes) # Get label of related URI
            else: # Blank node, etc.
                obj_str = str(o)

            fact_str = f"{subject_label}{type_info_str} - {prop_label}: {obj_str}"
            if fact_str not in retrieved_facts:
                 retrieved_facts.append(fact_str)
                 logger.debug(f"Added other fact: {fact_str}")
                 other_props_count +=1

            if other_props_count >= max_other_props_per_subject or len(retrieved_facts) >= max_facts:
                break

        subjects_processed_count += 1
        if len(retrieved_facts) >= max_facts:
            logger.info(f"Reached max_facts ({max_facts}). Stopping fact retrieval.")
            break

    logger.info(f"Retrieved {len(retrieved_facts)} relevant facts for question: '{question[:100]}...'")
    if retrieved_facts:
        logger.debug("First few retrieved facts:")
        for i, fact_item in enumerate(retrieved_facts[:3]):
            logger.debug(f"  Fact {i+1}: {fact_item}")

    return retrieved_facts[:max_facts] # Ensure we don't exceed max_facts


if __name__ == '__main__':
    # Configure basic console logging for direct script testing
    # This setup is for when running this file directly.
    # The "pipeline_trace" logger is primarily for when used within the Streamlit app.
    if not logger.handlers: # Avoid adding handlers if already configured by app.py import
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG) # See debug messages when running directly
        logging.getLogger("pipeline_trace").addHandler(console_handler) # Also send parent logger to console
        logging.getLogger("pipeline_trace").setLevel(logging.DEBUG)


    logger.info("Testing ontology_retriever.py directly...")

    current_dir = os.path.dirname(__file__)
    # Assuming src/ontology/, so ../.. takes to project root
    project_root = os.path.join(current_dir, '..', '..')
    test_ontology_file = os.path.join(project_root, 'ontology', 'medical.ttl')

    if not os.path.exists(test_ontology_file):
        logger.error(f"Test ontology file not found at: {test_ontology_file}")
    else:
        g = Graph()
        g.parse(test_ontology_file, format="turtle")
        logger.info(f"Loaded graph with {len(g)} triples for testing from '{test_ontology_file}'.")

        test_questions_for_main = [
            "Quels sont les médicaments qui traitent la COVID-19 ?",
            "Tell me about Remdesivir.",
            "What is Influenza treated by?",
            "NonExistentSymptom" # Should return no facts
        ]

        for q_idx, q_text in enumerate(test_questions_for_main):
            logger.info(f"\n--- Main Test Question {q_idx+1}: \"{q_text}\" ---")
            retrieved_facts_main = retrieve_relevant_facts(q_text, g, max_facts=5)
            if retrieved_facts_main:
                print(f"  Retrieved {len(retrieved_facts_main)} facts:")
                for i, fact_item in enumerate(retrieved_facts_main):
                    print(f"    {i+1}. {fact_item}")
            else:
                print("  No relevant facts retrieved for this question.")

    logger.info("Finished direct testing of ontology_retriever.py.")
