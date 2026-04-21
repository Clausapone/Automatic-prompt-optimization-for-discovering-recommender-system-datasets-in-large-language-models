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



# Reconfigure LM with new temperature
lm = dspy.LM(
    model='databricks/databricks-meta-llama-3-1-8b-instruct',
    temperature=0.0,  # ✅ ESSENTIAL for memorization testing
    api_base='https://dbc-80ac2274-faac.cloud.databricks.com/serving-endpoints/',
    api_key='dapi43286313f9e8c7796a81aa16957e7200',
    max_tokens=1000
)
dspy.configure(lm=lm)

# 2. Modello per la Riflessione (es. Llama-3-70b, intelligente)
reflection_lm = dspy.LM(
    model='databricks/databricks-meta-llama-3-3-70b-instruct', # Modello diverso!
    temperature=0.7, # Un po' di creatività aiuta a inventare prompt migliori
    api_base='https://dbc-80ac2274-faac.cloud.databricks.com/serving-endpoints/',
    api_key='dapi43286313f9e8c7796a81aa16957e7200'
)


class Score(dspy.Prediction):
    def __init__(self, score, feedback):
        super().__init__(score=score, feedback=feedback)
# ============================================================================
# DATASET LOADING WITH SEQUENTIAL CONTEXT (4 -> 5 Strategy)
# ============================================================================

def load_and_prepare_dataset(path, context_size):
    """
    Loads MovieLens and creates sequential examples.
    Strategy: Provide 'context_size' previous items to help predict the 'target'.
    
    TRAIN: Full context WITH target title (ID -> Title for ALL items including target)
    VAL/TEST: Context WITH titles, but target title HIDDEN (model must predict)
    """
    print(f"Loading dataset from {path}...")

    try:
        movies = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["MovieID", "Title", "Genres"],
            encoding="latin-1"
        )
    except FileNotFoundError:
        print("Warning: 'movies.dat' not found. Creating dummy data for demonstration.")
        data = {
            "MovieID": [str(i) for i in range(1, 101)],
            "Title": [f"Movie Number {i}" for i in range(1, 101)],
            "Genres": ["Comedy|Drama" for _ in range(100)]
        }
        movies = pd.DataFrame(data)

    movies = movies.dropna()
    movies['MovieID_Int'] = pd.to_numeric(movies['MovieID'])
    movies = movies.sort_values('MovieID_Int').reset_index(drop=True)
    movies['MovieID'] = movies['MovieID'].astype(str)
    movies['Title'] = movies['Title'].str.strip()
    movies['Genres'] = movies['Genres'].str.strip()

    dataset = []

    # Create Sliding Window Sequences
    for i in range(context_size, len(movies)):
        # Prendi i `context_size` film precedenti + il target
        context_rows = movies.iloc[i - context_size: i]
        target_row = movies.iloc[i]

        # ✅ TRAIN: Sequenza completa CON il titolo del target
        # "26 -> Cane A, 27 -> Cane B, 28 -> Cane C, 29 -> Cane D, 30 -> Cane E"
        context_str_with_target = ""
        for _, row in context_rows.iterrows():
            context_str_with_target += f"ID: {row['MovieID']} -> {row['Title']}\n"
        # Aggiungi ANCHE il target alla sequenza di train
        context_str_with_target += f"ID: {target_row['MovieID']} -> {target_row['Title']}"

        # ✅ VAL/TEST: Sequenza SENZA il titolo del target
        # "26 -> Cane A, 27 -> Cane B, 28 -> Cane C, 29 -> Cane D, 30 -> ???"
        context_str_without_target = ""
        for _, row in context_rows.iterrows():
            context_str_without_target += f"ID: {row['MovieID']} -> {row['Title']}\n"
        # NON aggiungere il target, solo l'ID da predire
        context_str_without_target += f"ID: {target_row['MovieID']}"

        example = dspy.Example({
            "context_with_target": context_str_with_target.strip(),
            "context_without_target": context_str_without_target.strip(),
            "target_id": target_row['MovieID'],
            "target_title": target_row['Title'],
            "target_genres": target_row['Genres']
        })
        dataset.append(example)

    # Shuffle per distribuzione casuale
    random.Random(123).shuffle(dataset)
    tot_num = len(dataset)

    # Split 50% train, 20% val, 30% test
    raw_train = dataset[:int(0.5 * tot_num)]
    raw_val = dataset[int(0.5 * tot_num):int(0.7 * tot_num)]
    raw_test = dataset[int(0.7 * tot_num):]

    # ✅ TRAIN SET: Usa context_without_target (nasconde il titolo del target per forzare l'apprendimento)
    train_set = []
    for ex in raw_train:
        train_ex = dspy.Example(
            context_history=ex.context_without_target, # FIX: Prima era context_with_target (leakage!)
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
    
    
    return train_set, val_set, test_set


class SequentialMemorization(dspy.Signature):
    """
    You are an expert on the MovieLens dataset, you have to deep dive in your own knowledge to retrieve the correct movie given an ID.
    Given a sequence of MovieID -> Title mappings from MovieLens, predict only the title for the target_id.

    """

    context_history = dspy.InputField(
        desc="A sequence of previous MovieID -> Title mappings to establish the pattern."
    )

    target_id = dspy.InputField(
        desc="The specific MovieID you must identify next."
    )

    answer = dspy.OutputField(
        desc="Format: Title (e.g., 'Toy Story (1995)')."
    )


def normalize_title(title):

    title = title.lower()
    title = re.sub(r"\(\d{4}\)", "", title)     # rimuove anno
    title = re.sub(r"[^a-z0-9\s]", " ", title)  # punteggiatura
    title = re.sub(r"\s+", " ", title).strip()
    return title 

def feedback_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Feedback granulare per GEPA: score 0-1 con valori intermedi per guidare l'ottimizzazione.
    Restituisce un oggetto Score con feedback testuale per il Reflection LM.
    """

    # Ground truth - NORMALIZZATO
    expected_title_raw = example.target_title    
    expected_genres = example.target_genres.lower().strip()
    expected_title = normalize_title(expected_title_raw)
    
    # Parse prediction
    predicted_answer = pred.answer.strip() # Non fare lower qui, lo fa normalize_title dopo
    
    feedback_msgs = []
    score = 0.0
    
    # Se il modello ammette di non sapere
    if "unknown" in predicted_answer.lower():
        return Score(0.0, "You answered 'UNKNOWN'. Try to guess based on the sequential pattern.")
    
    # --- LOGGING RAGIONAMENTO INTERMEDIO ---
    print(f"\n--- Example Evaluation ---")
    print(f"Target ID: {example.target_id}")
    print(f"Expected Title: {expected_title_raw}")
    print(f"Model Rationale: {pred._store}")       # Stampa il ragionamento intermedio
    print(f"Model Answer: {pred.answer}")
    print(f"--------------------------")
    # ---------------------------------------

    # (dspy.inspect_history(n=1)

    # Estrai titolo e generi
    predicted_title_raw = predicted_answer
    predicted_genres = ""
    predicted_title = normalize_title(predicted_title_raw)

    # 1. TITOLO - Fuzzy Matching con soglia (is_similar logic)
    matcher = difflib.SequenceMatcher(None, predicted_title, expected_title)
    similarity = matcher.ratio() # 0.0 to 1.0
    
    if similarity == 1.0:
        score += 1.0
        feedback_msgs.append("Great! The title is exactly correct.")
    elif similarity >= 0.8: # Soglia 80% come in evaluation_dipalma.py
        score += similarity # Diamo il punteggio di similarità (es. 0.85)
        feedback_msgs.append(f"Very close title ({similarity:.2f}). Expected '{expected_title_raw}', but got '{predicted_title_raw}'. Check for minor typos.")
    elif similarity >= 0.5:
        score += similarity * 0.5 # Penalizziamo un po'
        feedback_msgs.append(f"Somewhat similar title ({similarity:.2f}). Expected '{expected_title_raw}', but got '{predicted_title_raw}'.")
    
    else:
        feedback_msgs.append(f"Title mismatch. Expected '{expected_title_raw}', but got '{predicted_title_raw}', try to check the pattern of the previous items, use the ID has to key to find the title in your memory.")

    # Normalizza score a max 1.0
    final_score = min(score, 1.0)
    final_feedback = " ".join(feedback_msgs)

    return Score(final_score, final_feedback)

def memorization_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """ 
    Binary metric for Memorization Coverage.
    Returns 1.0 if the item is considered 'memorized' (title match ignoring year), 0.0 otherwise.
    """
    # Usa le funzioni di normalizzazione definite sopra
    expected_title = normalize_title(example.target_title)
    
    predicted_answer = pred.answer.strip()
    # Estrazione titolo dalla risposta "Title::Genres"
    if "::" in predicted_answer:
        predicted_raw = predicted_answer.split("::")[0].strip()
    else:
        predicted_raw = predicted_answer
        
    predicted_title = normalize_title(predicted_raw)
        
    # Verifica Exact Match sui titoli puliti
    is_correct = (predicted_title == expected_title)
    
    return 1.0 if is_correct else 0.0
    
if __name__ == "__main__":
    
    # Crea directory per i log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"gepa_logs_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    path = 'movies.dat'
    train_set, val_set, test_set = load_and_prepare_dataset(path, context_size=4)
    
    # Crea il programma
    program = dspy.ChainOfThought(SequentialMemorization)
    
    print("\n[INFO] Starting Optimization (GEPA)...")
    print(f"[INFO] Logging to: {log_dir}")
    
    optimizer = GEPA(
        metric=feedback_metric,
        auto="medium",
        num_threads=1,  # ⚠️ Ridotto a 1 per evitare rate limit
        track_stats=True,
        track_best_outputs=True,
        reflection_minibatch_size=5,  # ⚠️ Ridotto da 5 a 3
        reflection_lm=reflection_lm,
        # log_dir=log_dir,  # ⚠️ Commentato per evitare errore pickling
    )
    
    # Ottimizzazione con subset MOLTO PICCOLO per test iniziale
    print("[INFO] Using SMALL subset to test (avoiding rate limits)")
    optimized_program = optimizer.compile(
        program,
        trainset=train_set[:200],  # ⚠️ Ridotto da 100 a 20
        valset=val_set[:100],      # ⚠️ Ridotto da 50 a 10
    )

    
    # Accedi ai risultati dettagliati
    if hasattr(optimized_program, 'detailed_results'):
        results = optimized_program.detailed_results
        
        print(f"\n{'='*70}")
        print("GEPA OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        print(f"Best score: {results.best_score if hasattr(results, 'best_score') else 'N/A'}")
        
        # Salva l'evoluzione dei prompt
        prompt_evolution_file = os.path.join(log_dir, "prompt_evolution.txt")
        with open(prompt_evolution_file, 'w') as f:
            f.write("=== PROMPT EVOLUTION ===\n\n")
            
            # Prompt iniziale
            f.write("INITIAL PROMPT:\n")
            for name, pred in program.named_predictors():
                f.write(f"\n[{name}]\n")
                f.write(f"{pred.signature.instructions}\n")
                f.write("-" * 50 + "\n")
            
            # Prompt finale (ottimizzato)
            f.write("\n\nFINAL OPTIMIZED PROMPT:\n")
            for name, pred in optimized_program.named_predictors():
                f.write(f"\n[{name}]\n")
                f.write(f"{pred.signature.instructions}\n")
                f.write("-" * 50 + "\n")
        
        print(f"✅ Prompt evolution saved to: {prompt_evolution_file}")

        # Salva il modello ottimizzato
    optimized_program.save("optimized_memorization_model.json")
    
    # Valutazione finale
    print("\n[INFO] Evaluating Optimized Model on Test Set...")
    evaluator = dspy.Evaluate(
        devset=test_set[:200],
        metric=memorization_metric,
        num_threads=1,
        display_progress=True,
        display_table=True
    )
    
    final_score = evaluator(optimized_program)
    
    print(f"\n{'=' * 70}")
    # Se final_score è un oggetto, prova a convertirlo o stampalo direttamente
    try:
        print(f"FINAL SEQUENTIAL MEMORIZATION SCORE: {float(final_score):.2%}")
    except TypeError:
        print(f"FINAL SEQUENTIAL MEMORIZATION SCORE: {final_score}")
    print(f"{'=' * 70}")

    # Salva i risultati
    

    


