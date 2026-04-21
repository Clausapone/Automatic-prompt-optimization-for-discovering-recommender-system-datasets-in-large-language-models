import os
from datetime import datetime
import dspy
import pandas as pd
import random
from local_gepa.gepa import GEPA
import re
from typing import List, Dict, Tuple
import time
from functools import wraps
from litellm import api_base
import logging
import difflib
import json
from test_metrics import adaptive_similarity

# ============================================================================
# LOGGING CONFIGURATION - TESTING WITHOUT USING SEQUENCING
# ============================================================================
def setup_logging(log_dir):
    """
    Configura il sistema di logging con file handlers multipli per diversi livelli di dettaglio.
    
    Returns:
        tuple: (main_logger, execution_logger, prompt_logger, metrics_logger)
    """
    # Crea directory per i log se non esiste
    os.makedirs(log_dir, exist_ok=True)
    
    # Formatter dettagliato
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Formatter semplice
    simple_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    
    # ===== MAIN LOGGER (tutto) =====
    main_logger = logging.getLogger('GEPA_Main')
    main_logger.setLevel(logging.DEBUG)
    
    main_file_handler = logging.FileHandler(os.path.join(log_dir, 'main_execution.log'))
    main_file_handler.setFormatter(detailed_formatter)
    main_logger.addHandler(main_file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    main_logger.addHandler(console_handler)
    
    # ===== EXECUTION LOGGER (fasi di esecuzione) =====
    exec_logger = logging.getLogger('GEPA_Execution')
    exec_logger.setLevel(logging.INFO)
    
    exec_file_handler = logging.FileHandler(os.path.join(log_dir, 'execution_phases.log'))
    exec_file_handler.setFormatter(detailed_formatter)
    exec_logger.addHandler(exec_file_handler)
    
    # ✅ AGGIUNGI CONSOLE OUTPUT
    exec_console_handler = logging.StreamHandler()
    exec_console_handler.setLevel(logging.INFO)
    exec_console_handler.setFormatter(simple_formatter)
    exec_logger.addHandler(exec_console_handler)
    
    # ===== PROMPT EVOLUTION LOGGER =====
    prompt_logger = logging.getLogger('GEPA_Prompt')
    prompt_logger.setLevel(logging.INFO)
    
    prompt_file_handler = logging.FileHandler(os.path.join(log_dir, 'prompt_evolution.log'))
    prompt_file_handler.setFormatter(detailed_formatter)
    prompt_logger.addHandler(prompt_file_handler)
    
    # ✅ AGGIUNGI CONSOLE OUTPUT (solo per messaggi importanti)
    prompt_console_handler = logging.StreamHandler()
    prompt_console_handler.setLevel(logging.WARNING)  # Solo warning+ per non intasare
    prompt_console_handler.setFormatter(simple_formatter)
    prompt_logger.addHandler(prompt_console_handler)
    
    # ===== METRICS LOGGER =====
    metrics_logger = logging.getLogger('GEPA_Metrics')
    metrics_logger.setLevel(logging.INFO)
    
    metrics_file_handler = logging.FileHandler(os.path.join(log_dir, 'metrics_tracking.log'))
    metrics_file_handler.setFormatter(detailed_formatter)
    metrics_logger.addHandler(metrics_file_handler)
    
    # ✅ AGGIUNGI CONSOLE OUTPUT
    metrics_console_handler = logging.StreamHandler()
    metrics_console_handler.setLevel(logging.INFO)
    metrics_console_handler.setFormatter(simple_formatter)
    metrics_logger.addHandler(metrics_console_handler)
    
    # Evita la propagazione ai logger parent
    for logger in [main_logger, exec_logger, prompt_logger, metrics_logger]:
        logger.propagate = False
    
    return main_logger, exec_logger, prompt_logger, metrics_logger

# Reconfigure LM with new temperature - sostituire con i modelli su SGLANG (3,1-8B, 3,2-3B, e poi usare l'api da 70B)
lm = dspy.LM(
    model='databricks/databricks-meta-llama-3-3-70b-instruct',
    temperature=0.0,  # ✅ ESSENTIAL for memorization testing
    api_base='https://dbc-80ac2274-faac.cloud.databricks.com/serving-endpoints/',
    api_key='dapi43286313f9e8c7796a81aa16957e7200',
    max_tokens=1000
)

# 2. Modello per la Riflessione (es. Llama-3-70b, intelligente)
reflection_lm = dspy.LM(
    model='databricks/databricks-meta-llama-3-3-70b-instruct',
    temperature=0.7, # Un po' di creatività aiuta a inventare prompt migliori
    api_base='https://dbc-80ac2274-faac.cloud.databricks.com/serving-endpoints/',
    api_key='dapi43286313f9e8c7796a81aa16957e7200',
    max_tokens=1000,
    num_retries=10  # Aumenta i tentativi per gestire il rate limit
)

dspy.configure(lm=lm, reflection_lm=reflection_lm)


class Score(dspy.Prediction):
    def __init__(self, score, feedback):
        super().__init__(score=score, feedback=feedback)
# ============================================================================
# DATASET LOADING WITH SEQUENTIAL CONTEXT (4 -> 5 Strategy)
# ============================================================================

def load_and_prepare_dataset_interactions(path):
    """
    Loads MovieLens interactions and creates examples for next-interaction prediction.
    SEQUENTIAL STRATEGY:
    - Group by UserID
    - Sort by Timestamp
    - For each user sequence [I1, I2, I3...], create pairs:
      Input (I_t) -> Predict (I_t+1)
    
    The dataset split (Train/Val/Test) is done on the *interactions* globally, 
    preserving the order (e.g., first 70% of total interactions for train),
    OR better, we can split by users? 
    Usually, for sequential recommendation, we might do "leave-last-out" or time-based split.
    Here, to keep it simple and consistent with previous logic:
    We collect ALL valid pairs (Current -> Next) from all users, then split.
    """
    print(f"Loading dataset from {path}...")
    try:
        # Assuming MovieLens ratings.dat format: UserID::MovieID::Rating::Timestamp
        interactions = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["UserID", "MovieID", "Rating", "Timestamp"],
            encoding="latin-1"
        )
    except FileNotFoundError:
        print("Warning: Dataset not found. Creating dummy data.")
        # Dummy data
        interactions = pd.DataFrame({
            "UserID": [1, 1, 1, 2, 2],
            "MovieID": [101, 102, 103, 201, 202],
            "Rating": [5, 4, 5, 3, 4],
            "Timestamp": [100, 101, 102, 200, 201]
        })

    # Sort interactions by UserID and Timestamp to guarantee correct sequence
    interactions = interactions.sort_values(by=['UserID', 'Timestamp']).reset_index(drop=True)
    
    dataset = []

    # Group by UserID to ensure we only predict next interaction WITHIN the same user session
    grouped = interactions.groupby('UserID')

    for user_id, group in grouped:
        # group is a DataFrame for a single user, sorted by time
        if len(group) < 2:
            continue
            
        group = group.reset_index(drop=True)
        # Create sliding pairs
        for i in range(len(group) - 1):
            current_row = group.iloc[i]
            next_row = group.iloc[i+1]
            
            # Input: ID::MovieID
            current_interaction = f"{current_row['UserID']}::{current_row['MovieID']}"
            # Target: ID::NextMovieID
            next_interaction = f"{next_row['UserID']}::{next_row['MovieID']}"
            
            example = dspy.Example({
                "user_id": str(current_row['UserID']),
                "movie_id": str(current_row['MovieID']),
                "next_interaction": next_interaction
            }).with_inputs("user_id", "movie_id")
            
            dataset.append(example)

    # Split Train/Val/Test (Time-ordered split of the collected examples is tricky because they are mixed users)
    # We'll do a simple split of the list. Since the list is ordered by User (1..N), 
    # train will have users 1..X, test users Y..Z. This is a "User Split" strategy essentially.
    
    tot_num = len(dataset)
    split_1 = int(0.7 * tot_num)
    split_2 = int(0.85 * tot_num)

    raw_train = dataset[:split_1]
    raw_val = dataset[split_1:split_2]
    raw_test = dataset[split_2:]
    
    # Helper to clean/finalize examples
    def finalize_examples(raw_list):
        out = []
        for ex in raw_list:
            out.append(dspy.Example(
                user_id=ex.user_id,
                movie_id=ex.movie_id,
                next_interaction=ex.next_interaction
            ).with_inputs("user_id", "movie_id"))
        return out

    train_set = finalize_examples(raw_train)
    val_set = finalize_examples(raw_val)
    test_set = finalize_examples(raw_test)

    print(f"\nSequential Interaction Dataset Created (Grouped by User)")
    print(f"Total pairs: {tot_num}")
    print(f"Train: {len(train_set)}")
    print(f"Val: {len(val_set)}")
    print(f"Test: {len(test_set)}")
    
    # Debug sample
    if len(train_set) > 0:
        print("[DEBUG Sample] Input:", train_set[0].user_id, "::", train_set[0].movie_id, 
              "-> Target:", train_set[0].next_interaction)

    return train_set, val_set, test_set


def load_and_prepare_dataset_items(path):
    """
    Loads MovieLens and creates sequential examples.
    Strategy: Provide 'context_size' previous items to help predict the 'target'.
    
    TRAIN: Full context WITH target title (ID -> Title for ALL items including target)
    VAL/TEST: Context WITH titles, but target title HIDDEN (model must predict)
    """
    print(f"Loading dataset from {path}...")

    try:
        movies = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Warning: '{path}' not found. Creating dummy data for demonstration.")
        data = {
            "movieId": [str(i) for i in range(1, 101)],
            "title": [f"Movie Number {i}" for i in range(1, 101)],
            "genres": ["Comedy|Drama" for _ in range(100)]
        }
        movies = pd.DataFrame(data)

    movies = movies.dropna()
    movies['MovieID_Int'] = pd.to_numeric(movies['movieId'])
    movies = movies.sort_values('MovieID_Int').reset_index(drop=True)
    movies['MovieID'] = movies['movieId'].astype(str)
    movies['Title'] = movies['title'].str.strip()
    movies['Genres'] = movies['genres'].str.strip()

    dataset = []

    # Create Sliding Window Sequences
    for i in range(context_size, len(movies)):
        # Prendi i `context_size` film precedenti + il target
        context_rows = movies.iloc[i - context_size: i]
        target_row = movies.iloc[i]

        # ✅ TRAIN: Sequenza completa CON il titolo del target
        context_str_with_target = ""
        for _, row in context_rows.iterrows():
            context_str_with_target += f"ID: {row['MovieID']} -> {row['Title']} ({row['Genres']})\n"
        # Aggiungi ANCHE il target alla sequenza di train
        context_str_with_target += f"ID: {target_row['MovieID']} -> {target_row['Title']} ({target_row['Genres']})"

        # ✅ VAL/TEST: Sequenza SENZA il titolo del target
        context_str_without_target = ""
        for _, row in context_rows.iterrows():
            context_str_without_target += f"ID: {row['MovieID']} -> {row['Title']} ({row['Genres']})\n"
        # NON aggiungere il titolo del target
        context_str_without_target += f"ID: {target_row['MovieID']}"

        example = dspy.Example({
            "context_with_target": context_str_with_target.strip(),
            "context_without_target": context_str_without_target.strip(),
            "target_id": target_row['MovieID'],
            "target_title": target_row['Title'],
            "target_genres": target_row['Genres']
        })
        dataset.append(example)

  
    tot_num = len(dataset)
    split_1 = int(0.8 * tot_num)
    split_2 = int(0.9 * tot_num)

    # Split Sequenziale (ID 1-XXX Train, XXX-YYY Val, YYY-ZZZ Test)
    raw_train = dataset[:split_1]
    raw_val = dataset[split_1:split_2]
    raw_test = dataset[split_2:]

    # ✅ TRAIN SET: Usa context_without_target (nasconde il titolo del target per forzare l'apprendimento)
    train_set = []
    for ex in raw_train:
        train_ex = dspy.Example(
            context_history=ex.context_without_target, 
            target_id=ex.target_id,
            target_title=ex.target_title,
            target_genres=ex.target_genres
        ).with_inputs("context_history", "target_id")
        train_set.append(train_ex)

    # ✅ VAL SET: Usa context_without_target (nasconde il titolo del target)
    val_set = []
    for ex in raw_val:
        val_ex = dspy.Example(
            context_history=ex.context_without_target,
            target_id=ex.target_id,
            target_title=ex.target_title,
            target_genres=ex.target_genres
        ).with_inputs("context_history", "target_id")
        val_set.append(val_ex)

    # ✅ TEST SET: Usa context_without_target (nasconde il titolo del target)
    test_set = []
    for ex in raw_test:
        test_ex = dspy.Example(
            context_history=ex.context_without_target,
            target_id=ex.target_id,
            target_title=ex.target_title,
            target_genres=ex.target_genres
        ).with_inputs("context_history", "target_id")
        test_set.append(test_ex)

    print(f"\nSequential Dataset Created. Window Size: {context_size}")
    print(f"Train: {len(train_set)} (WITH target title for learning)")
    print(f"Val: {len(val_set)} (WITHOUT target title - must predict)")
    print(f"Test: {len(test_set)} (WITHOUT target title - must predict)")

    # --- DEBUG VERIFICATION ---
    print("\n" + "="*70)
    print("DEBUG: VERIFYING DATASET CONTENT")
    print("="*70)
    
    if len(train_set) > 0:
        print("\n--- [TRAIN EXAMPLE SAMPLE] ---")
        ex = train_set[0]
        print(f"INPUT (Context History):\n{ex.context_history}")
        print("-" * 20)
        print(f"INPUT (Target ID): {ex.target_id}")
        print(f"GOLD LABEL (Target Title): {ex.target_title}")
        print("Note: In TRAIN, we expect the context to end WITHOUT the target title, forcing prediction.")

    if len(test_set) > 0:
        print("\n--- [TEST EXAMPLE SAMPLE] ---")
        ex = test_set[0]
        print(f"INPUT (Context History):\n{ex.context_history}")
        print("-" * 20)
        print(f"INPUT (Target ID): {ex.target_id}")
        print(f"GOLD LABEL (Target Title): {ex.target_title}")
        print("Note: In TEST, the target title MUST BE HIDDEN from the input.")
    
    print("="*70 + "\n")
    
    return train_set, val_set, test_set

def load_and_prepare_dataset_user(path):
    """
    Carica il dataset users.dat di MovieLens e crea esempi strutturati per la memorizzazione.
    
    Formato: UserID::Gender::Age::Occupation::Zip-code
    
    Informazioni demografiche:
    - Gender: "M" (male) o "F" (female)
    - Age: Codici categorici (1, 18, 25, 35, 45, 50, 56)
        * 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44"
        * 45: "45-49", 50: "50-55", 56: "56+"
    - Occupation: Codici da 0 a 20 (es. 0: "other", 12: "programmer", 14: "sales/marketing")
    - Zip-code: Codice postale USA
    
    Strategia:
    - INPUT: UserID
    - OUTPUT: Gender::Age::Occupation::Zip-code
    - No sequenze (per ora), solo lookup diretto UserID -> attributi
    
    Parameters:
        path (str): Percorso del file users.dat
        
    Returns:
        tuple: (train_set, val_set, test_set) - liste di dspy.Example
    """
    print(f"Loading dataset from {path}...")
    
    try:
        users = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
            encoding="latin-1"
        )
    except FileNotFoundError:
        print("Warning: 'users.dat' not found. Creating dummy data for demonstration.")
        data = {
            "UserID": [str(i) for i in range(1, 101)],
            "Gender": ["M" if i % 2 == 0 else "F" for i in range(1, 101)],
            "Age": [25 for _ in range(100)],
            "Occupation": [12 for _ in range(100)],
            "Zip-code": ["12345" for _ in range(100)]
        }
        users = pd.DataFrame(data)
    
    # ========================================================================
    # DATA CLEANING & NORMALIZATION
    # ========================================================================
    users = users.dropna()
    users['UserID_Int'] = pd.to_numeric(users['UserID'])
    users = users.sort_values('UserID_Int').reset_index(drop=True)
    users['UserID'] = users['UserID'].astype(str)
    
    # Normalizza i campi
    users['Gender'] = users['Gender'].str.strip()
    users['Age'] = users['Age'].astype(str).str.strip()
    users['Occupation'] = users['Occupation'].astype(str).str.strip()
    users['Zip-code'] = users['Zip-code'].astype(str).str.strip()
    
    # ========================================================================
    # CREATE STRUCTURED EXAMPLES
    # ========================================================================
    dataset = []
    
    for _, row in users.iterrows():
        # Crea la stringa target (tutto tranne l'UserID)
        target_attributes = f"{row['Gender']}::{row['Age']}::{row['Occupation']}::{row['Zip-code']}"
        
        example = dspy.Example({
            "user_id": row['UserID'],
            "gender": row['Gender'],
            "age": row['Age'],
            "occupation": row['Occupation'],
            "zipcode": row['Zip-code'],
            "target_attributes": target_attributes  # Formato completo per il ground truth
        })
        dataset.append(example)
    
    # ========================================================================
    # TRAIN/VAL/TEST SPLIT
    # ========================================================================
    tot_num = len(dataset)
    split_1 = int(0.7 * tot_num)
    split_2 = int(0.85 * tot_num)
    split_3 = int(1 * tot_num)
    
    # Split Sequenziale (UserID 1-XXX Train, XXX-YYY Val, YYY-ZZZ Test)
    raw_train = dataset[:split_1]
    raw_val = dataset[split_1:split_2]
    raw_test = dataset[split_2:split_3]
    
    # ✅ TRAIN SET: Input = UserID, Output = Gender::Age::Occupation::Zip-code
    train_set = []
    for ex in raw_train:
        train_ex = dspy.Example(
            user_id=ex.user_id,
            target_attributes=ex.target_attributes,
            gender=ex.gender,
            age=ex.age,
            occupation=ex.occupation,
            zipcode=ex.zipcode
        ).with_inputs("user_id")
        train_set.append(train_ex)
    
    # ✅ VAL SET
    val_set = []
    for ex in raw_val:
        val_ex = dspy.Example(
            user_id=ex.user_id,
            target_attributes=ex.target_attributes,
            gender=ex.gender,
            age=ex.age,
            occupation=ex.occupation,
            zipcode=ex.zipcode
        ).with_inputs("user_id")
        val_set.append(val_ex)
    
    # ✅ TEST SET
    test_set = []
    for ex in raw_test:
        test_ex = dspy.Example(
            user_id=ex.user_id,
            target_attributes=ex.target_attributes,
            gender=ex.gender,
            age=ex.age,
            occupation=ex.occupation,
            zipcode=ex.zipcode
        ).with_inputs("user_id")
        test_set.append(test_ex)
    
    print(f"\nUser Dataset Created (No Sequences)")
    print(f"Train: {len(train_set)} examples")
    print(f"Val: {len(val_set)} examples")
    print(f"Test: {len(test_set)} examples")
    
    # ========================================================================
    # DEBUG VERIFICATION
    # ========================================================================
    print("\n" + "="*70)
    print("DEBUG: VERIFYING USER DATASET CONTENT")
    print("="*70)
    
    if len(train_set) > 0:
        print("\n--- [TRAIN EXAMPLE SAMPLE] ---")
        ex = train_set[0]
        print(f"INPUT (UserID): {ex.user_id}")
        print("-" * 20)
        print(f"EXPECTED OUTPUT: {ex.target_attributes}")
        print(f"  - Gender: {ex.gender}")
        print(f"  - Age: {ex.age}")
        print(f"  - Occupation: {ex.occupation}")
        print(f"  - Zip-code: {ex.zipcode}")
        print("Note: Model must predict Gender::Age::Occupation::Zip-code from UserID")
    
    if len(test_set) > 0:
        print("\n--- [TEST EXAMPLE SAMPLE] ---")
        ex = test_set[0]
        print(f"INPUT (UserID): {ex.user_id}")
        print("-" * 20)
        print(f"EXPECTED OUTPUT: {ex.target_attributes}")
        print(f"  - Gender: {ex.gender}")
        print(f"  - Age: {ex.age}")
        print(f"  - Occupation: {ex.occupation}")
        print(f"  - Zip-code: {ex.zipcode}")
        print("Note: This is a pure memorization task (ID -> Attributes lookup)")
    
    print("="*70 + "\n")
    
    return train_set, val_set, test_set

# essendo basato sugli ID nuovamente è possibile espploraare le sequenze
class SequentialMemorization_user(dspy.Signature):
    '''You are the Movielens dataset.
    When given a lookup key (e.g., a UserID), you will respond wit the exact corresponding Gender,Age,Occupation,Zip-code value from the dataset.
    Below an example of query and its correct response, do not include the ID in the output:
    M::50::14::13110
    '''
    
    user_id = dspy.InputField(
        desc="The specific UserID provided from the Movielens dataset"
    )

    answer = dspy.OutputField(
        desc="The interaction string ONLY (e.g., 'M::50::14::13210') "
    )

class SequentialMemorization_item(dspy.Signature):
    "You are the Movielens dataset. "
    "When given a lookup key (e.g., a MovieID), you will respond with the exact corresponding value from the dataset. "
    

    context_history = dspy.InputField(
        desc="A sequence of previous MovieID -> Title mappings."
    )

    target_id = dspy.InputField(
        desc="The specific MovieID you must identify next."
    )

    answer = dspy.OutputField(
        desc="The movie title ONLY (e.g., 'Toy Story (1995)'). No extra text."
    )

class SequentialMemorization_interaction(dspy.Signature):
    """
    You are a Recommender System.
    Given a UserID and a MovieID representing the current interaction, predict the NEXT interaction (UserID::MovieID) in the sequence.
    The sequence is continuous across the dataset.
    """
    
    user_id = dspy.InputField(desc="The UserID of the current interaction")
    movie_id = dspy.InputField(desc="The MovieID of the current interaction")
    
    answer = dspy.OutputField(desc="The next interaction string in format 'UserID::MovieID'")


def normalize_title(title):
    title = title.lower().strip()
    title = re.sub(r"\(\d{4}\)", "", title)     # rimuove anno
    title = re.sub(r"[^a-z0-9\s]", " ", title)  # punteggiatura
    title = re.sub(r"\s+", " ", title).strip()
    return title 

def feedback_metric_item(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Feedback granulare per GEPA: score 0-1 con valori intermedi e 'aiuti' (cheats) per l'ottimizzazione.
    """

    # Ground truth
    expected_title_raw = example.target_title    
    expected_title = normalize_title(expected_title_raw)
    
    # Prediction
    predicted_answer = pred.answer.strip()
    predicted_title = normalize_title(predicted_answer)
    
    feedback_msgs = []
    score = 0.0
    
    # 1. Unknown Penalty
    if "unknown" in predicted_answer.lower():
        return Score(0.0, f"Don't give up! Try to output a movie title matching ID {example.target_id}. The correct title was '{expected_title_raw}'.")

    # 2. Fuzzy Matching
    matcher = difflib.SequenceMatcher(None, predicted_title, expected_title)
    similarity = matcher.ratio()
 

    if similarity == 1.0:
        score = 1.0
        feedback_msgs.append("Perfect match!")
    elif similarity >= 0.7:
        score = similarity + 0.1 # Boost per essere andati molto vicini
        feedback_msgs.append(f"Almost perfect ({similarity}). Expected '{expected_title_raw}' vs Answer '{predicted_answer}'.")
    elif similarity >= 0.4:
        score = similarity
        feedback_msgs.append(f"Good guess but not exact ({similarity}). You said '{predicted_answer}', expected '{expected_title_raw}'.")
    else:
        score = 0.0
        feedback_msgs.append(f"Wrong title. ID {example.target_id} corresponds to '{expected_title_raw}'. Memorize this mapping.")

    final_score = min(score, 1.0)
    final_feedback = " ".join(feedback_msgs)
    
    # Debug print (opzionale, per non intasare troppo rimuoviamo la stampa massiva a ogni passo)

    return Score(final_score, final_feedback)

def memorization_metric_item(example, pred, trace=None, pred_name=None, pred_trace=None):

    expected_title = normalize_title(example.target_title)
    
    predicted_answer = pred.answer.strip()
    if "::" in predicted_answer:
        predicted_raw = predicted_answer.split("::")[0].strip()
    else:
        predicted_raw = predicted_answer
        
    predicted_title = normalize_title(predicted_raw)
        
    # Verifica Fuzzy
    matcher = difflib.SequenceMatcher(None, predicted_title, expected_title)
    similarity = matcher.ratio()
    
    # Successo se: Similarità >= 70% OPPURE Sottostringa valida
    return 1.0 if (similarity >= 0.7) else 0.0
    
    # Successo se: Similarità >= 70% OPPURE Sottostringa valida
    return 1.0 if (similarity >= 0.7) else 0.0

def feedback_metric_interaction(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Feedback for interaction prediction (UserID::MovieID).
    Example standard: UserID::MovieID
    """
    expected = example.next_interaction.strip()
    predicted = pred.answer.strip()
    
    # Exact match logic or fuzzy logic
    if expected == predicted:
        return Score(1.0, "Perfect match!")
    
    # Check if format is correct
    if "::" not in predicted:
        return Score(0.0, f"Format error. Expected 'UserID::MovieID', got '{predicted}'.")

    # Similarity
    matcher = difflib.SequenceMatcher(None, predicted, expected)
    similarity = matcher.ratio()
    
    if similarity >= 0.8:
        return Score(similarity, f"Very close! Expected '{expected}', got '{predicted}'.")
    else:
        return Score(similarity, f"Wrong interaction. Expected '{expected}'.")

def memorization_metric_interaction(example, pred, trace=None, pred_name=None, pred_trace=None):
    expected = example.next_interaction.strip()
    predicted = pred.answer.strip()
    return 1.0 if (expected == predicted) else 0.0

def feedback_metric_user(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Feedback granulare per GEPA: score 0-1 con valori intermedi e 'aiuti' (cheats) per l'ottimizzazione.
    Checks similarity between predicted attributes and target attributes.
    """

    # Ground truth
    target = example.target_attributes
    
    # Prediction
    predicted_answer = pred.answer
    
    feedback_msgs = []
    score = 0.0
    
    # 1. Unknown Penalty
    if "unknown" in predicted_answer.lower():
        return Score(0.0, f"Don't give up! Try to output attributes matching UserID {example.user_id}. The correct attributes were '{target}'.")

    similarity_2, feedback_2 = adaptive_similarity(target, predicted_answer, feedback_msgs)

    if similarity_2 >= 0.7:
        return Score(1, feedback_2)
    else:
        return Score(similarity_2, f'Not enough for passing the minimal threshold. Try again! {feedback_2}')

def memorization_metric_user(example, pred, trace=None, pred_name=None, pred_trace=None):
    target = example.target_attributes
    predicted_answer = pred.answer
    
    feedback_msgs = []  # Non serve qui, ma adaptive_similarity lo richiede
    similarity, _ = adaptive_similarity(target, predicted_answer, feedback_msgs)
    
    # Successo se: Similarità >= 70%
    return 1.0 if (similarity >= 0.7) else 0.0

if __name__ == "__main__":
    
    # ========================================================================
    # SETUP & INITIALIZATION
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"gepa_logs_{timestamp}"
    
    # Setup logging system
    main_logger, exec_logger, prompt_logger, metrics_logger = setup_logging(log_dir)
    
    main_logger.info("="*80)
    main_logger.info("GEPA OPTIMIZATION RUN STARTED")
    main_logger.info("="*80)
    main_logger.info(f"Timestamp: {timestamp}")
    main_logger.info(f"Log Directory: {log_dir}")
    
    # ========================================================================
    # DATASET LOADING
    # ========================================================================
    exec_logger.info("PHASE 1: Loading and preparing dataset...")
    path_item = 'datasets/movielens100k_movies.csv'
    path_user = 'users.dat'
    path_interaction = 'ratings.dat'
    context_size = 14
    
    '''try:
        train_set, val_set, test_set = load_and_prepare_dataset(path_item, context_size=context_size)
        exec_logger.info(f"✅ Dataset loaded successfully")
        exec_logger.info(f"   - Train set: {len(train_set)} examples")
        exec_logger.info(f"   - Validation set: {len(val_set)} examples")
        exec_logger.info(f"   - Test set: {len(test_set)} examples")
        exec_logger.info(f"   - Context size: {context_size}")
    except Exception as e:
        main_logger.error(f"❌ Failed to load dataset: {e}")
        raise'''
    
    train_set, val_set, test_set = load_and_prepare_dataset_items(path_item)
    # train_set, val_set, test_set = load_and_prepare_dataset_user(path_user)
    # train_set, val_set, test_set = load_and_prepare_dataset_interactions("ratings.dat")
    
    # ========================================================================
    # PROGRAM INITIALIZATION
    # ========================================================================
    exec_logger.info("PHASE 2: Initializing program and optimizer...")
    
    # Using Chain of Thought program
    program_item = dspy.ChainOfThought(SequentialMemorization_item)
    
    '''# Configurazione optimizer for items
    optimizer_config = {
        "metric": feedback_metric_item,
        "auto": "medium",
        "num_threads": 1,
        "track_stats": True,
        "track_best_outputs": True,
        "reflection_minibatch_size": 10,
    }'''
    
    optimizer = GEPA(
        metric=feedback_metric_item,
        auto="medium",
        num_threads=8,  
        track_stats=True,
        track_best_outputs=True,
        reflection_minibatch_size=10,  
        reflection_lm=reflection_lm,
    )
    
    exec_logger.info("✅ Optimizer initialized")
    
    # ========================================================================
    # OPTIMIZATION PHASE
    # ========================================================================
    exec_logger.info("PHASE 3: Starting GEPA optimization...")
    metrics_logger.info("Starting optimization tracking...")
    
    start_time = time.time()
    
    try:
        optimized_program = optimizer.compile(
            program_item,
            trainset=train_set,  
            valset=val_set,      
        )
        
        optimization_time = time.time() - start_time
        exec_logger.info(f"✅ Optimization completed in {optimization_time:.2f} seconds")
        metrics_logger.info(f"Total optimization time: {optimization_time:.2f}s")
        
    except Exception as e:
        main_logger.error(f"❌ Optimization failed: {e}")
        exec_logger.error(f"Optimization error: {e}")
        raise

    # ========================================================================
    # SAVE OPTIMIZATION ARTIFACTS
    # ========================================================================
    exec_logger.info("PHASE 4: Saving optimization artifacts...")
    
    # Accedi ai risultati dettagliati
    if hasattr(optimized_program, 'detailed_results'):
        results = optimized_program.detailed_results
        
        main_logger.info("="*80)
        main_logger.info("GEPA OPTIMIZATION RESULTS")
        main_logger.info("="*80)
        
        if hasattr(results, 'best_score'):
            best_score = results.best_score
            main_logger.info(f"Best score during optimization: {best_score}")
            metrics_logger.info(f"Best validation score: {best_score}")
        
        # Salva l'evoluzione dei prompt
        prompt_evolution_file = os.path.join(log_dir, "prompt_evolution.txt")
        with open(prompt_evolution_file, 'w') as f:
            f.write("=== PROMPT EVOLUTION ===\n\n")
            # Prompt finale (ottimizzato)
            f.write("\n\nFINAL OPTIMIZED PROMPT:\n")
            for name, pred in optimized_program.named_predictors():
                f.write(f"\n[{name}]\n")
                f.write(f"{pred.signature.instructions}\n")
                f.write("-" * 50 + "\n")
        
        exec_logger.info(f"✅ Prompt evolution saved to: {prompt_evolution_file}")

    # Salva il modello ottimizzato
    model_file = os.path.join(log_dir, "optimized_memorization_model.json")
    optimized_program.save(model_file)
    exec_logger.info(f"✅ Optimized model saved to: {model_file}")
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    exec_logger.info("PHASE 5: Final evaluation on test set...")
    metrics_logger.info("Starting final evaluation...")
    
    # Evaluate memorization - da cambiare a seconda del setup (items, users, item-user interaction)
    evaluator = dspy.Evaluate(
        devset=test_set,
        metric=memorization_metric_item,
        num_threads=1,
        display_progress=True,
        display_table=True
    )
    
    eval_start_time = time.time()
    final_score = evaluator(optimized_program)
    eval_time = time.time() - eval_start_time
    
    exec_logger.info(f"✅ Evaluation completed in {eval_time:.2f} seconds")
    
    # Log final score
    try:
        score_value = float(final_score)
        main_logger.info(f"FINAL SEQUENTIAL MEMORIZATION SCORE: {score_value:.2%}")
        metrics_logger.info(f"Final test score: {score_value:.2%}")
        metrics_logger.info(f"Evaluation time: {eval_time:.2f}s")
    except TypeError:
        main_logger.info(f"FINAL SEQUENTIAL MEMORIZATION SCORE: {final_score}")
        metrics_logger.info(f"Final test score: {final_score}")
    
    main_logger.info("="*80)
    
    # ========================================================================
    # SAVE COMPREHENSIVE RESULTS
    # ========================================================================
    exec_logger.info("PHASE 6: Saving comprehensive results...")
    

    # Salva i risultati
    def save_optimization_results(optimized_program, final_score, test_set, log_dir, timestamp):
        """
        Salva tutti i risultati dell'ottimizzazione GEPA in modo strutturato.
        
        Args:
            optimized_program: Il programma ottimizzato da GEPA
            final_score: Score finale della valutazione
            test_set: Dataset di test usato
            log_dir: Directory dove salvare i risultati
            timestamp: Timestamp per identificare l'esecuzione
        """
        import json
        
        results_file = os.path.join(log_dir, "optimization_results.txt")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("GEPA OPTIMIZATION RESULTS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # 1. Informazioni generali
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Test Set Size: {len(test_set)}\n")
            
            # 2. Score finale
            try:
                score_value = float(final_score)
                f.write(f"Final Memorization Score: {score_value}\n")
            except (TypeError, ValueError):
                f.write(f"Final Memorization Score: {final_score}\n")
            
            f.write("\n" + "-"*80 + "\n\n")
            
            # 3. Dettagli del programma ottimizzato
            f.write("OPTIMIZED PROGRAM STRUCTURE:\n")
            f.write("-"*80 + "\n")
            for name, pred in optimized_program.named_predictors():
                f.write(f"\nPredictor: {name}\n")
                f.write(f"Signature: {pred.signature.__class__.__name__}\n")
                if hasattr(pred.signature, 'instructions'):
                    f.write(f"Instructions:\n{pred.signature.instructions}\n")
                f.write("\n")
            
            f.write("\n" + "-"*80 + "\n\n")
            
            # 4. Statistiche di ottimizzazione (se disponibili)
            if hasattr(optimized_program, 'detailed_results'):
                f.write("OPTIMIZATION DETAILS:\n")
                f.write("-"*80 + "\n")
                results = optimized_program.detailed_results
                
                if hasattr(results, 'best_score'):
                    f.write(f"Best Score Durante Ottimizzazione: {results.best_score}\n")
                
                if hasattr(results, 'iterations'):
                    f.write(f"Numero di Iterazioni: {results.iterations}\n")
                    
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("="*80 + "\n")
        
        exec_logger.info(f"✅ Results summary saved to: {results_file}")
        main_logger.info(f"Results summary written to: {results_file}")
        
        # Salva anche un JSON con i dati strutturati per analisi successive
        json_file = os.path.join(log_dir, "results_data.json")
        results_data = {
            "timestamp": timestamp,
            "test_set_size": len(test_set),
            "final_score": str(final_score),
            "log_directory": log_dir
        }
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            exec_logger.info(f"✅ JSON data saved to: {json_file}")
        except Exception as e:
            main_logger.error(f"⚠️ Error saving JSON: {e}")
        
        return results_file
    
    # Chiama la funzione per salvare i risultati
    save_optimization_results(optimized_program, final_score, test_set, log_dir, timestamp)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    total_runtime = time.time() - start_time
    
    main_logger.info("="*80)
    main_logger.info("GEPA OPTIMIZATION RUN COMPLETED SUCCESSFULLY")
    main_logger.info("="*80)
    main_logger.info(f"Total runtime: {total_runtime} seconds ({total_runtime/60} minutes)")
    main_logger.info(f"All results saved to: {log_dir}")
    main_logger.info("")
    main_logger.info("Generated files:")
    main_logger.info(f"  - main_execution.log       : Complete execution log")
    main_logger.info(f"  - execution_phases.log     : Phase-by-phase tracking")
    main_logger.info(f"  - prompt_evolution.log     : Prompt changes during optimization")
    main_logger.info(f"  - metrics_tracking.log     : Performance metrics")
    main_logger.info(f"  - prompt_evolution.txt     : Human-readable prompt comparison")
    main_logger.info(f"  - optimization_results.txt : Complete results summary")
    main_logger.info(f"  - results_data.json        : Structured data for analysis")
    main_logger.info(f"  - optimized_memorization_model.json : Trained model")
    main_logger.info("="*80)
    
    print(f"\n✅ Run completed! Check logs in: {log_dir}")
