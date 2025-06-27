# ==============================================================================
# FICHIER : src/ontology/graph_interrogator.py
# Rôle : Moteur de raisonnement pour le Mode 3. C'est le cœur de l'intelligence
#        de notre pipeline. Il identifie les concepts dans la question, cherche
#        les liens logiques entre eux dans l'ontologie, et prépare un rapport
#        d'analyse complet pour le LLM.
# ==============================================================================

# --- Imports des bibliothèques nécessaires ---
import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import deque  # Un type de liste optimisé pour ajouter/enlever des éléments au début/fin
from itertools import permutations # Un outil pour générer toutes les paires possibles d'entités
import re
from rdflib import Graph, URIRef, Literal, RDF, RDFS # Les outils pour manipuler le graphe

logger = logging.getLogger("pipeline_trace." + __name__)
# ==============================================================================
# FONCTION : _identify_all_entities
# Rôle : "Le Traducteur". Comprendre la question de l'utilisateur en la liant
#        aux concepts techniques précis de notre ontologie.
# ==============================================================================
def _identify_all_entities(question: str, graph: Graph) -> List[URIRef]:
    """
    Identifie tous les concepts (entités) de l'ontologie qui sont mentionnés
    dans la question de l'utilisateur. C'est la porte d'entrée du raisonnement.
    Elle utilise une approche hybride pour être à la fois précise et intelligente.
    """
    # On commence par écrire dans les logs que cette étape débute.
    logger.info("Étape 3A : Début de l'identification d'entités (méthode hybride)...")
    
    # On prépare un 'set' (un ensemble sans doublons) pour stocker les entités que l'on va trouver.
    found_entities = set()
    
    # On met la question de l'utilisateur en minuscules pour que la recherche ne soit pas sensible à la casse.
    question_lower = question.lower()

    # --- PARTIE 1 : RECHERCHE PAR SYNONYMES (Pour comprendre le langage courant) ---
    # Cette partie sert de "traducteur" entre les mots simples de l'utilisateur et les termes techniques de l'ontologie.
    
    # On définit notre dictionnaire de synonymes.
    # À gauche (la clé) : un mot simple que l'utilisateur pourrait taper.
    # À droite (la valeur) : une liste de noms de concepts techniques (labels) qui correspondent à ce mot.
    synonym_map = {
        "solution": ["Algorithme de compensation", "Stratégie d'Optimisation", "Padding mémoire 3D"],
        "corriger": ["Algorithme de compensation", "Stratégie d'Optimisation"],
        "lent": ["Goulot d'étranglement réseau", "Haute latence inter-groupe"],
        "problème": ["Problème de précision", "ProblemePerformance", "False sharing", "Bank conflicts L2"]
    }
    
    # On parcourt chaque entrée de notre dictionnaire de synonymes.
    for keyword, labels_in_ontology in synonym_map.items():
        # Si un mot-clé (ex: "solution") est présent dans la question de l'utilisateur...
        if keyword in question_lower:
            # ...alors on parcourt la liste des labels techniques associés (ex: ["Algorithme de compensation", "Stratégie d'Optimisation"]).
            for label_text in labels_in_ontology:
                # Pour chaque label, on demande au graphe : "Quel est le sujet qui a pour label exact ce texte ?".
                # La fonction graph.value(...) fait cette recherche très efficacement.
                subject = graph.value(predicate=RDFS.label, object=Literal(label_text))
                
                # Si on a trouvé un sujet (un concept) et que c'est bien une URI...
                if subject and isinstance(subject, URIRef):
                    # ...on l'ajoute à notre liste de trouvailles !
                    logger.info(f"Entité trouvée via synonyme ('{keyword}' -> '{label_text}'): {subject}")
                    found_entities.add(subject)

    # --- PARTIE 2 : RECHERCHE DIRECTE (Pour les termes techniques) ---
    # C'est une sécurité pour trouver les concepts dont le nom exact est déjà dans la question.
    
    # On parcourt tous les faits de type "label" dans l'ontologie.
    # Le `(None, RDFS.label, None)` signifie "donne-moi tous les triplets où le prédicat est rdfs:label".
    for s, p, o in graph.triples((None, RDFS.label, None)):
        # On s'assure de travailler avec des données propres.
        if isinstance(s, URIRef) and isinstance(o, Literal):
            # On prend le label de l'ontologie (ex: "MPI_Alltoall") et on le met en minuscules.
            label_lower = str(o.value).lower()
            
            # Si ce label est trouvé quelque part dans la question...
            if label_lower in question_lower:
                # ...et qu'on ne l'a pas déjà ajouté via un synonyme...
                if s not in found_entities:
                    logger.info(f"Entité trouvée par label direct : '{o.value}' -> {s}")
                    # ...alors on l'ajoute à nos résultats.
                    found_entities.add(s)

    # --- Étape Finale : On retourne le résultat ---
    if not found_entities:
        logger.warning("Aucune entité n'a été trouvée.")
    else:
        logger.info(f"{len(found_entities)} entité(s) identifiée(s).")
        
    # On convertit notre ensemble de résultats en une liste et on la retourne.
    # C'est cette liste qui sera utilisée par la suite du raisonnement.
    return list(found_entities)





def _get_node_label(node_uri: URIRef, graph: Graph) -> str:
    """Fonction utilitaire pour 'traduire' un identifiant technique (URI) en nom lisible."""
    if not isinstance(node_uri, URIRef): return str(node_uri)
    label = graph.value(subject=node_uri, predicate=RDFS.label)
    if label: return str(label.value)
    # Si pas de label, on nettoie l'URI pour la rendre plus courte.
    return node_uri.split('#')[-1].split('/')[-1]



# ==============================================================================
# FONCTION : _find_shortest_path
# Rôle : "L'Explorateur" ou le "GPS". Trouve le chemin le plus court
#        entre deux concepts dans le graphe.
# ==============================================================================
def _find_shortest_path(graph: Graph, start_node: URIRef, end_node: URIRef) -> Optional[List[Tuple[URIRef, URIRef, URIRef]]]:
    """
    Trouve le plus court chemin entre deux concepts dans le graphe. C'est l'algorithme de DÉDUCTION.
    Il utilise une méthode de parcours en largeur (Breadth-First Search ou BFS).
    """
    # Cas simple : si le point de départ est le même que l'arrivée, le chemin est vide.
    if start_node == end_node: return []
    
    # On initialise la recherche.
    # La "queue" est une liste de tâches à faire. Au début, il n'y a qu'une tâche : explorer le point de départ.
    # Chaque élément de la queue est une paire : (le_noeud_a_explorer, le_chemin_deja_parcouru_pour_y_arriver).
    queue = deque([(start_node, [])]) 
    
    # "visited" est notre "mémoire" pour ne pas explorer deux fois le même endroit et tourner en rond.
    visited = {start_node} 

    # Tant qu'il y a des tâches dans la liste d'attente...
    while queue:
        # ...on prend la plus ancienne tâche (c'est le principe du "premier arrivé, premier servi").
        current_node, path = queue.popleft() 
        
        # --- PHASE 1 : Explorer les voisins "sortants" ---
        # On cherche toutes les flèches qui PARTENT du nœud actuel.
        for row in graph.predicate_objects(subject=current_node):
            p, o = row[0], row[1] # p = prédicat (la flèche), o = objet (la destination)
            # Si la destination est un autre concept et qu'on ne l'a jamais visité...
            if isinstance(o, URIRef) and o not in visited:
                # ...on met à jour le chemin : on ajoute ce nouveau "pas".
                new_path = path + [(current_node, p, o)] 
                # Si cette destination est notre cible finale, BINGO ! On a trouvé le chemin.
                if o == end_node: return new_path 
                # Sinon, on marque ce voisin comme "visité" et on l'ajoute à la liste des tâches à faire.
                visited.add(o)
                queue.append((o, new_path))
        
        # --- PHASE 2 : Explorer les voisins "entrants" (pour les liens inverses) ---
        # On fait la même chose pour les flèches qui ARRIVENT au nœud actuel.
        for row in graph.subject_predicates(object=current_node):
             s, p = row[0], row[1] # s = sujet (le point de départ de la flèche)
             if isinstance(s, URIRef) and s not in visited:
                new_path = path + [(s, p, current_node)]
                if s == end_node: return new_path # On a trouvé la cible !
                visited.add(s)
                queue.append((s, new_path))
                
    # Si la boucle se termine (plus aucune tâche à faire) et qu'on n'a pas trouvé la cible,
    # c'est qu'il n'existe aucun chemin. On retourne "None".
    return None

# ==============================================================================
# FONCTION : find_reasoning_path
# Rôle : Orchestrateur du raisonnement du Mode 3.
# ==============================================================================
def find_reasoning_path(question: str, graph: Graph) -> Dict:
    """
    Fonction principale du Mode 3. Elle orchestre l'identification des concepts,
    la déduction des liens logiques et la collecte des faits pertinents.
    """

    # --- ÉTAPE 1 : IDENTIFICATION DES CONCEPTS ---
    # La toute première chose à faire est de comprendre de quoi parle la question.
    # On appelle notre fonction d'identification intelligente `_identify_all_entities` (que nous avons déjà vue).
    # Le résultat est une liste des concepts de notre ontologie trouvés dans la question.
    # Exemple: pour "lien entre MPI_Alltoall et bande passante", initial_entities contiendra
    # les concepts :MPI_Alltoall et :BandePassanteReseau.
    initial_entities = _identify_all_entities(question, graph)

    # Si on n'a trouvé aucun concept, le raisonnement est impossible.
    # On retourne immédiatement un rapport vide pour éviter les erreurs.
    if not initial_entities:
        return {"question": question, "entites_initiales": [], "chemin_trouve": [], "faits_deduits": []}
    
    # --- ÉTAPE 2 : LA DÉCISION STRATÉGIQUE ---
    # C'est ici que le "Stratège" prend sa décision la plus importante.

    # On prépare des variables pour stocker nos découvertes.
    path_found = None          # Stockera le chemin logique s'il en existe un.
    all_facts_triplets = set() # Stockera tous les faits bruts (triplets) que nous trouverons.

    # On active la logique de recherche de chemin uniquement si on a trouvé AU MOINS 2 concepts.
    # C'est logique : il faut au moins un point de départ ET un point d'arrivée pour chercher un chemin.
    if len(initial_entities) >= 2:
        # On utilise `permutations` pour tester toutes les paires possibles de concepts.
        # Par exemple, si on a trouvé A et B, on va chercher un chemin de A vers B, PUIS de B vers A.
        for start_node, end_node in permutations(initial_entities, 2):
            # On appelle notre "explorateur" `_find_shortest_path` pour faire la recherche.
            path_found = _find_shortest_path(graph, start_node, end_node)
            # Si un chemin est trouvé, on arrête de chercher. Le premier est souvent le plus pertinent.
            if path_found:
                break 
    
    # --- ÉTAPE 3 : LA COLLECTE DES FAITS PERTINENTS ---
    # Maintenant, on décide quelles informations on va mettre dans notre rapport final,
    # en fonction de la décision prise à l'étape 2.

    # CAS A : On a trouvé un chemin logique entre les concepts.
    if path_found:
        # Le raisonnement est considéré comme une réussite. Le contexte sera très ciblé.
        # On ajoute tous les triplets (les "pas") du chemin trouvé à notre collection de faits.
        all_facts_triplets.update(path_found)

        # Pour rendre le rapport plus clair, on ajoute aussi les informations de base
        # (le nom et le type) de chaque concept qui fait partie du chemin.
        nodes_in_path = {s for s,p,o in path_found} | {o for s,p,o in path_found if isinstance(o, URIRef)}
        for node in nodes_in_path:
            for s_fact, p_fact, o_fact in graph.triples((node, None, None)):
                if p_fact in [RDFS.label, RDF.type]:
                    all_facts_triplets.add((s_fact, p_fact, o_fact))
    
    # CAS B : On n'a pas trouvé de chemin, ou il n'y avait qu'un seul concept au départ.
    else: 
        # On se rabat sur une stratégie plus simple : on explore le "voisinage" du concept le plus pertinent.
        if initial_entities:
            main_entity = initial_entities[0] # On prend le premier concept de la liste.
            # On ajoute simplement tous les faits directement liés à ce concept (toutes les flèches qui en partent).
            for s, p, o in graph.triples((main_entity, None, None)):
                all_facts_triplets.add((s,p,o))

    # --- ÉTAPE 4 : FORMATAGE DU RAPPORT FINAL ---
    # On transforme nos découvertes techniques en un rapport lisible.

    # 1. On transforme les faits bruts (triplets) en phrases lisibles.
    formatted_facts = [f"{_get_node_label(s, graph)} --[{_get_node_label(p, graph)}]--> {_get_node_label(o, graph) if isinstance(o, URIRef) else f'\"{o}\"'}" for s, p, o in all_facts_triplets]
    
    # 2. On fait la même chose pour le chemin trouvé, s'il y en a un.
    formatted_path = [f"{_get_node_label(s, graph)} --[{_get_node_label(p, graph)}]--> {_get_node_label(o, graph)}" for s, p, o in path_found] if path_found else []

    # 3. On retourne un dictionnaire Python qui contient toutes ces informations bien organisées.
    return {
        "question": question,
        "entites_initiales": [_get_node_label(e, graph) for e in initial_entities],
        "chemin_trouve": formatted_path,
        "faits_deduits": formatted_facts
    }