import os
import logging
import pandas as pd
import random
import numpy as np
import torch
import openai
from tqdm import tqdm
from difflib import SequenceMatcher
import re
import yaml
import time
from openai import OpenAI
import time
import requests 
import json 
import dspy
from litellm import api_base 

# Setting the logging - tracking the execution
logging.basicConfig(level=logging.INFO, format='\n%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Integration with Databricks - setting the class
class SequentialUser(dspy.Signature):
    ''' 
    You are tasked with generating an interaction string based on a given user ID from the Movielens dataset. The Movielens dataset contains demographic information about users, including their gender, age, occupation, and zip code. 

    To perform this task, you will need to look up the user ID in the Movielens dataset and retrieve the corresponding demographic information. The interaction string is structured as 'M::Age::Occupation::Zip-code' for male users and 'F::Age::Occupation::Zip-code' for female users.

    The age ranges in the dataset are: 
    1 (18-24), 
    18 (25-34), 
    25 (35-44), 
    35 (45-49), 
    50 (50-55), and 
    56 (56 or older).

    The occupation clusters in the dataset are: 
    0 (other or none), 
    1 (academic/educator), 
    2 (artist), 
    3 (clerical/admin), 
    4 (college/grad student), 
    5 (customer service), 
    6 (doctor/health care), 
    7 (executive/managerial), 
    8 (farmer), 
    9 (homemaker), 
    10 (K-12 student), 
    11 (lawyer), 
    12 (programmer), 
    13 (retired), 
    14 (sales/marketing), 
    15 (scientist), 
    16 (self-employed), 
    17 (technician/engineer), 
    18 (tradesman/craftsman), 
    19 (unemployed), and 
    20 (writer).

    The zip codes in the dataset are 5-digit codes.

    Given a user ID, you will respond with the exact corresponding Gender, Age, Occupation, and Zip-code value from the dataset, excluding the ID in the output, and following the specified format and using the correct age ranges and occupation clusters.

    If the user ID is not found in the dataset, you should not provide any response.

    Please respond with the correct interaction string for a given user ID.
    '''

    user_id = dspy.InputField(
        desc="The specific UserID provided from the Movielens dataset"
    )

    answer = dspy.OutputField(
        desc="The interaction string ONLY (e.g., 'M::50::14::13210') "
    )

def calculate_age_similarity(age_real, age_pred, feedback, score):
    AGE_RANGES = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+"
    } 
    # Ordine dei range
    age_order = [1, 18, 25, 35, 45, 50, 56]

    # Funzione per mappare un'età al cluster corrispondente (lower bound)
    def map_to_cluster(age):
        # Prende il valore più alto in age_order che è <= age
        # Es: 20 -> 18, 30 -> 25
        candidates = [x for x in age_order if x <= age]
        return max(candidates) if candidates else 1

    # Mappa valori reali e predetti ai rispettivi cluster
    cluster_real = map_to_cluster(age_real)
    cluster_pred = map_to_cluster(age_pred)

    # Trova le posizioni nell'ordine usando i cluster mappati
    pos_real = age_order.index(cluster_real)
    pos_pred = age_order.index(cluster_pred)

    # Distanza ordinale (numero di "salti" tra range)
    distance = abs(pos_real - pos_pred)
    max_distance = len(age_order) - 1  # 6 nel nostro caso

    # Normalizza: 0 salti = 1.0, max salti = 0.0
    age_similarity = 1 - (distance / max_distance)

    score['age_score'] = age_similarity * 0.30

    return score, feedback

def calculate_occupation_similarity_semantic(occ_real, occ_pred, feedback, score):
    """
    Match esatto = 1.0
    Stesso cluster semantico = 0.3
    Diverso = 0.0
    """

    # Gruppi semantici di lavori correlati
    OCCUPATION_CLUSTERS = {
        'education': [1, 4, 10],           # academic, student, K-12
        'creative': [2, 20],                # artist, writer
        'tech': [12, 15, 17],              # programmer, scientist, engineer
        'healthcare': [6],                  # doctor
        'business': [7, 14, 16],           # executive, sales, self-employed
        'service': [3, 5],                  # clerical, customer service
        'manual': [8, 18],                  # farmer, tradesman
        'legal': [11],                      # lawyer
        'home': [9, 13, 19],               # homemaker, retired, unemployed
        'other': [0]
    }

    if occ_real == occ_pred:
        feedback.append('The occupation is correct')
        score['occupation_score'] = 0.2
        return score, feedback

    # Trova i cluster
    cluster_real = None
    cluster_pred = None

    if occ_pred != occ_real:
        for cluster_name, occupations in OCCUPATION_CLUSTERS.items():
            if occ_real in occupations:
                cluster_real = cluster_name
            if occ_pred in occupations:
                cluster_pred = cluster_name

        # Stesso cluster = credito parziale
        if cluster_real == cluster_pred and cluster_real is not None:
            feedback.append('The occupation belong to the right cluster')
            score['occupation_score'] = 0.1
            return score, feedback
        else:
            feedback.append('The occupation is incorrect and doesnt belong to any cluster, try to follow the cluster')
            score['occupation_score'] = 0

            return score, feedback

def calculate_postal_similarity(postal_real, postal_pred, feedback, score):
    """
    Similarità per codici postali con enfasi sulle prime cifre (area geografica).

    Combina:
    1. Prefix matching con pesi decrescenti (area geografica)
    2. Levenshtein normalizzata per errori parziali
    3. Bonus per lunghezza comune

    Args:
        postal_real: codice postale reale (es. "53706")
        postal_pred: codice postale predetto (es. "53122")

    Returns:
        float: similarità tra 0 (completamente diversi) e 1 (identici)
    """
    postal_real = str(postal_real).strip()
    postal_pred = str(postal_pred).strip()


    # === COMPONENTE 1: PREFIX MATCHING CON PESI GEOGRAFICI ===
    # Pesi decrescenti esponenzialmente per ogni posizione
    position_weights = [0.70, 0.15, 0.10, 0.025, 0.05]

    scores = 0.0
    # Confronta solo fino alla lunghezza dei pesi (5 cifre)
    for i, (r, p) in enumerate(zip(postal_real, postal_pred)):
        if i >= len(position_weights):
            break
        if r == p:
            scores += position_weights[i]

    score['postal_code_score'] = scores * 0.20
    return score, feedback


# Adaptive Similarity - metrica che si adatta alla struttura degli ID
def adaptive_similarity(real_string, pred_string, feedback):
    """
    Metrica che si adatta alla struttura degli ID
    """
    real_parts = real_string.split("::")
    pred_parts = pred_string.split("::")

    # Clean real_parts: remove ID because we want to compare attributes
    # real_string format is defined in fetch_user_attribute_with_LLM as:
    # UserID::Gender::Age::Occupation::Zip-code
    if len(real_parts) == 5:
        real_parts = real_parts[1:] # Now: [Gender, Age, Occupation, Zip-code]

    # Clean pred_parts: remove ID if present
    # Expected attributes: Gender, Age, Occupation, Zip-code
    if len(pred_parts) == 5:
         pred_parts = pred_parts[1:] # Assume first is ID, remove it
    elif len(pred_parts) != 4:
         # If it's not 5 (with ID) and not 4 (without ID), structure is likely wrong
         feedback.append('The format of the answer is incorrect, should be Gender::Age::Occupation::Zip-code')
         return 0, feedback

    score = {}

    # Calcola similarità sugli altri campi
    try:
        # Index 0: Gender
        if pred_parts[0] == real_parts[0]:
            feedback.append('The gender is correct')
            score['gender_score'] = 0.30 # Using weight consistent with previous code's intent
        else:
            feedback.append('The gender is incorrect')
            score['gender_score'] = 0

        # Index 1: Age (int)
        try:
            p_age = int(pred_parts[1])
            r_age = int(real_parts[1])
            score, feedback  = calculate_age_similarity(p_age, r_age, feedback, score)
        except ValueError:
            feedback.append(f'Invalid Age format: {pred_parts[1]}')
            score['age_score'] = 0

        # Index 2: Occupation (int)
        try:
            p_occ = int(pred_parts[2])
            r_occ = int(real_parts[2])
            score, feedback = calculate_occupation_similarity_semantic(p_occ, r_occ, feedback, score)
        except ValueError:
             feedback.append(f'Invalid Occupation format: {pred_parts[2]}')
             score['occupation_score'] = 0

        # Index 3: Zip-code (string)
        postal_real = real_parts[3]
        postal_pred = pred_parts[3]
        score, feedback = calculate_postal_similarity(postal_real, postal_pred, feedback, score)
        
        total_score = sum(score.values())
        return total_score, feedback

    except Exception as e:
        print(f'Errore : {e}')
        feedback.append('General error in similarity calculation')
        return 0, feedback

def load_and_prepare_dataset_item(path):
    """
    Loads MovieLens and creates sequential examples.
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
        print("Warning: 'movies.dat' not found. Creating dummy data.")
        # Dummy data creation omitted for brevity, assuming file exists as per previous context
        return [], [], []

    return movies

def load_and_prepare_dataset_user(path):
    """
    Loads MovieLens and creates sequential examples.
    """
    try:
        users = pd.read_csv(
            path,
        )
    except FileNotFoundError:
        print("Warning: 'users.dat' not found. Creating dummy data for demonstration.")

        return [], [], []

    return users

def compute_similarity(str_a, str_b):
    """
    Compute similarity (fuzzy match ratio) between two strings using rapidfuzz.

    Parameters:
        str_a (str): First string.
        str_b (str): Second string.

    Returns:
        float: Similarity ratio between 0 and 100.
    """
    ratio = SequenceMatcher(None, str_a, str_b).ratio()
    return round(ratio * 100, 2)

def query_databricks():

    lm = dspy.LM(
    model='databricks/databricks-meta-llama-3-1-8b-instruct',
    temperature=0.0,  # ✅ ESSENTIAL for memorization testing
    api_base='https://dbc-80ac2274-faac.cloud.databricks.com/serving-endpoints/',
    api_key='dapi43286313f9e8c7796a81aa16957e7200',
    max_tokens=1000
    )
    dspy.configure(lm=lm)
    return lm 

def query_sglang(sglang_pipeline, messages, model):
    """
    Query the SGlang with OpenAI compatible API

    Parameters:
        sglang_pipeline (object): The pipeline used to interact with the SGlang compatible OpenAI API.
        messages (list): A list of messages as dictionaries (e.g., {"role": "user", "content": "Your message"}).
        model (str): The model name (e.g., "meta-llama/Llama-3.1-405B-Instruct-FP8").

    Returns:
        str: Generated content from the model.
    """
    # Generate the completion
    completion = sglang_pipeline.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # Lower temperature for more focused responses
        top_p=1,  # Slightly higher for better fluency
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        n=1,  # Single response is usually more stable
        seed=42,  # Keep for reproducibility
    )
    print(completion.choices[0].message.content.strip())


    return completion.choices[0].message.content.strip()

def fetch_movie_name_with_LLM(items_df, config):
    """
    Given a DataFrame of movies with columns ["MovieID","Title"],
    use a language model to attempt to regenerate the movie title.

    Parameters:
        items_df (pd.DataFrame): DataFrame with "MovieID","Title" columns.
        config (dict): Configuration dictionary from YAML.

    Returns:
        pd.DataFrame: DataFrame with columns ["MovieID", "GeneratedTitle", "RealTitle", "ErrorFlag"].
    """

    # Prepare a lookup dict for faster title retrieval
    id_to_title = dict(zip(items_df['MovieID'], items_df['Title']))
    batch_size = 50
    sglang_pipeline = openai.Client(base_url="http://localhost:7501/v1", api_key="empty")
    results_file = f"{config['model_name'].replace('/', '_')}_results.csv"

    # Initialize or load existing results
    if os.path.exists(results_file):
        logger.info(f"Loading existing results from {results_file}.")
        existing_results = pd.read_csv(results_file)
        processed_ids = set(existing_results['MovieID'].astype(str))
        # Extract the last three examples for prompt
        last_three = existing_results.tail(3).to_dict('records')
    else:
        logger.info(f"Creating new results file: {results_file}.")
        existing_results = pd.DataFrame(columns=['MovieID', 'GeneratedTitle', 'RealTitle', 'ErrorFlag'])
        existing_results.to_csv(results_file, index=False)
        processed_ids = set()
        last_three = []  # No previous examples

    # Define initial examples for the first batch if no previous examples
    initial_examples = [
        {"MovieID": str(items_df.iloc[0]['MovieID']), "RealTitle": items_df.iloc[0]['Title']},
        {"MovieID": str(items_df.iloc[1]['MovieID']), "RealTitle": items_df.iloc[1]['Title']},
        {"MovieID": str(items_df.iloc[2]['MovieID']), "RealTitle": items_df.iloc[2]['Title']},
    ]

    # Prepare all movie IDs as strings
    movie_ids = items_df["MovieID"].astype(str).tolist()
    total_ids = len(movie_ids)


    # Progress bar for better tracking
    with tqdm(total=total_ids, desc="Processing Movies") as pbar:
        for batch_start in range(0, total_ids, batch_size):
            batch_end = min(batch_start + batch_size, total_ids)
            current_batch = movie_ids[batch_start:batch_end]

            # Filter out already processed MovieIDs
            current_batch = [mid for mid in current_batch if mid not in processed_ids]
            if not current_batch:
                pbar.update(batch_end - batch_start)
                continue

            tqdm.write(f"\nProcessing batch {batch_start + 1}-{batch_end} ({len(current_batch)} movies)...")

            # Select examples
            if last_three:
                examples = last_three
            else:
                # First batch: use initial_examples
                examples = initial_examples

            # Construct messages with selected examples
            messages = [
                {
                    "role": "system",
                    "content": (
                        prompt
                    )
                },
            ]

            for example in examples:
                messages.extend([
                    {
                        "role": "user",
                        "content": f"Input: {example['MovieID']}::"
                    },
                    {
                        "role": "assistant",
                        "content": f"{example['MovieID']}::{example['RealTitle']}\n"
                    },
                ])

            for movie_id_str in current_batch:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Input: {movie_id_str}::"
                    }
                )

                output = query_sglang(sglang_pipeline, messages, config['model_name'])
                if output == 'Unknown':
                    continue

                generated_title =  re.sub(r"\s*\(\d{4}\)$", "", re.sub(r"^\d+::\s*", "", output))
                real_title = id_to_title.get(int(movie_id_str), "Unknown")
                similarity = compute_similarity(generated_title, real_title)

                similarity_threshold = 70
                error_flag = 0 if similarity > similarity_threshold else 1
                if error_flag == 0:
                    logger.info(f"Correct - Generated: '{generated_title}' == Real: '{real_title}'")
                else:
                    logger.info(f"Error - Generated: '{generated_title}' <> Real: '{real_title}'")

                record = {
                    "MovieID": movie_id_str,
                    "GeneratedTitle": generated_title,
                    "RealTitle": real_title,
                    "ErrorFlag": error_flag
                }

                # Write the record immediately to the CSV file
                try:
                    pd.DataFrame([record]).to_csv(results_file, mode='a', header=False, index=False)
                except Exception as e:
                    logger.error(f"Failed to write record for MovieID {movie_id_str}: {e}")

                processed_ids.add(movie_id_str)

                # Update last_three examples
                if error_flag == 0:
                    last_three.append(record)
                    if len(last_three) > 3:
                        last_three.pop(0)
                else:
                    # Optionally, handle errors (e.g., retry, skip)
                    pass

                # Append the assistant's response to messages to maintain context
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"{movie_id_str}::{real_title}\n"
                    }
                )

            logger.info(f"Completed batch {batch_start + 1}-{batch_end}")
            pbar.update(batch_end - batch_start)

    logger.info("Processing completed.")
    # Load all results into a DataFrame before returning
    final_results = pd.read_csv(results_file)
    return final_results

def fetch_user_attribute_with_LLM(user_df,  config):
    """
    """
    # -------------------------------------------------------------------------
    # 1. Validate configuration
    # -------------------------------------------------------------------------
    dataset_name = config["dataset_name"]
    model_type = config["model_type"]
    batch_size = config["batch_size"]
    sglang_pipeline = openai.Client(base_url="http://localhost:7501/v1", api_key="fottiti")
    # results_file = f"{config['model_name'].replace('/', '_')}_results.csv"
    results_file = 'saved_user_output.csv'
    # -------------------------------------------------------------------------
    # TO - DO : filtering by models
    # -------------------------------------------------------------------------
    processed_rows = set()
    last_three = []  # No previous examples

    # Initialize or load existing results
    if os.path.exists(results_file):
        logger.info(f"Loading existing results from {results_file}.")
        existing_results = pd.read_csv(results_file)
        processed_ids = set(existing_results['UserID'].astype(str))
        
        # Extract and parse the last three examples for prompt
        last_three = []
        for _, row in existing_results.tail(3).iterrows():
            try:
                # RealUser format: UserID::Gender::Age::Occupation::Zip-code
                parts = str(row['RealUser']).split('::')
                if len(parts) == 5:
                    last_three.append({
                        "UserID": parts[0],
                        "Gender": parts[1],
                        "Age": parts[2],
                        "Occupation": parts[3],
                        "Zip-code": parts[4]
                    })
            except Exception:
                continue
    else:
        logger.info(f"Creating new results file: {results_file}.")
        existing_results = pd.DataFrame(columns=['UserID', 'RealUser', 'GeneratedUser', 'ErrorFlag'])
        existing_results.to_csv(results_file, index=False)
        processed_ids = set()
        last_three = []  # No previous examples


    # -------------------------------------------------------------------------
    # 4. Create interactions_row dict
    # -------------------------------------------------------------------------
    interactions_row = {}
    for i, row in user_df.iterrows():
        interactions_row[i] = f"{row['UserID']}::{row['Gender']}::{row['Age']}::{row['Occupation']}::{row['Zip-code']}"


    # -------------------------------------------------------------------------
    # 5. Define initial examples (few-shot) if no previous examples
    # -------------------------------------------------------------------------

    initial_examples = []
    max_rows_for_examples = min(3, len(user_df))
    for i in range(max_rows_for_examples):
        row = user_df.iloc[i]
        init_string = f"{row['UserID']}::{row['Gender']}::{row['Age']}::{row['Occupation']}::{row['Zip-code']}"
        example = {"RowIndex": i, "UserID":row['UserID'],
                   "Gender":row['Gender'], "Age":row['Age'],
                   "Occupation":row['Occupation'], "Zip-code":row['Zip-code'],
                   "InteractionString": init_string}
        initial_examples.append(example)

    # -------------------------------------------------------------------------
    # 6. Prepare row-based batching
    #    Instead of unique user IDs, we simply iterate over the row indices.
    # -------------------------------------------------------------------------
    total_rows = len(user_df)

    # spostare il prompt in maniera più pulita 

    prompt = """ 
    You are the Movielens dataset, a collection of movie ratings and demographic information. 
    Your task is to respond with the exact corresponding demographic information (Gender, Age, Occupation, Zip-code) for a given UserID.

    The input format will be a single number representing the UserID.

    The output format should be a string in the format "G::A::O::Z" where:
    - G is the Gender (M for Male, F for Female)
    - A is the Age (an integer between 1 and 100)
    - O is the Occupation (an integer representing one of 21 occupation clusters)
    - Z is the Zip-code (a 5-digit string)

    Note that the occupation clusters are numbered from 0 to 20, with the following mappings:
    - 0: other
    - 1: academic/educator
    - 2: artist
    - 3: clergy
    - 4: engineer
    - 5: entertainer
    - 6: executive
    - 7: healthcare professional
    - 8: homemaker
    - 9: lawyer
    - 10: librarian
    - 11: marketing
    - 12: musician
    - 13: programmer
    - 14: sales
    - 15: scientist
    - 16: student
    - 17: technician
    - 18: writer
    - 19: retired
    - 20: none

    The correct occupation cluster should be identified and used in the output.

    To improve the accuracy of your responses, consider the following:
    - Ensure the gender is correctly identified as either M (Male) or F (Female).
    - Verify the age is a valid integer between 1 and 100.
    - Use the correct occupation cluster based on the provided mappings.
    - Confirm the zip-code is a 5-digit string.

    Common mistakes to avoid include:
    - Incorrectly identifying the gender.
    - Using an invalid format for the age.
    - Misidentifying the occupation cluster.
    - Providing an incorrect zip-code.

    Based on the examples provided, it appears that the assistant has struggled with identifying the correct gender, occupation cluster, and zip-code for a given UserID. To improve performance, the assistant should carefully review the demographic information for each UserID and ensure that the output matches the exact information available in the Movielens dataset.

    Some specific examples of correct and incorrect responses are as follows:
    - For UserID 2797, the correct response is F::37::14::33210, not M::37::14::33210.
    - For UserID 4107, the correct response is F::39::0::13210, not M::39::14::13210.
    - For UserID 1546, the correct response is M::26::15::94125, but the occupation cluster may be incorrect.
    - For UserID 1102, the correct response is M::65::1::33150, not M::65::14::33150.


    The assistant should use a generalizable strategy to solve the task, such as:
    - Carefully reviewing the demographic information for each UserID.
    - Using the correct occupation cluster based on the provided mappings.
    - Ensuring the gender is correctly identified as either M (Male) or F (Female).
    - Verifying the age is a valid integer between 1 and 100.
    - Confirming the zip-code is a 5-digit string.

    By following these instructions and avoiding common mistakes, the assistant should be able to provide accurate demographic information for a given UserID.

    """


    # -------------------------------------------------------------------------------
    # 6. Client setup - setting the databricks setup using dspy - avoiding rate limit
    # -------------------------------------------------------------------------------
    
    def setup_databricks():
        lm = query_databricks()
        return lm

    def api_processing():
        
        program = dspy.Predict(SequentialUser)
        current_id = 1  # id che sto fornendo al modello generativo
        keys_id = 0     # chiave per accedere al corretto elemento del dizionario


        # -------------------------------------------------------------------------
        # 6. 2 - loop for testing the API
        # -------------------------------------------------------------------------

        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            # current_batch is a slice of interactions_df by row index
            current_batch = user_df.iloc[batch_start:batch_end]

            # Filter out already processed interactions
            current_batch = current_batch[~current_batch.index.isin(processed_rows)]

            if current_batch.empty:
                pbar.update(batch_end - batch_start)
                continue

            tqdm.write(f"\nProcessing batch {batch_start + 1}-{batch_end} ({len(current_batch)} rows)...")

            # 7c. Iterate through the rows in the current batch
            for i, row in current_batch.iterrows():
                if i in processed_rows:
                    # If already processed in a previous run, skip
                    pbar.update(1)
                    continue

                # Look up the "UserID" string
                # interaction = user_df.iloc[i]
                interaction = row

                result = program(user_id=current_id)
                logger.info(f"Result: {result}")
                generated_user = result.answer

                if generated_user == 'Unknown':
                    continue
                
                real_user = interactions_row[keys_id] 
                feedback = []
                similarity, _ = adaptive_similarity(generated_user, real_user, feedback)

                similarity = similarity * 100
                similarity_threshold = 70
                if similarity >= similarity_threshold:
                    error_flag = 0
                    logger.info(f"Correct - Generated: '{generated_user}' == Real: '{real_user}' score: {similarity}")
                else:
                    error_flag = 1
                    logger.info(f"Error - Generated: '{generated_user}' <> Real: '{real_user}' score: {similarity}")

                # ------------------------------------------------------------------
                # Save to CSV
                # ------------------------------------------------------------------
                record = {
                    "UserID": current_id,
                    "GeneratedUser": generated_user,
                    "RealUser": real_user,
                    "ErrorFlag": error_flag
                }
                try:
                    pd.DataFrame([record]).to_csv(results_file, mode='a', header=False, index=False)
                except Exception as e:
                    logger.error(f"Failed to write record for row {i}: {e}")

                processed_rows.add(i)

                # Proactive rate limiting
                time.sleep(1)
                current_id += 1
                keys_id += 1
                logger.info(f"Processed {i + 1} rows")

        logger.info(f"Completed batch {batch_start + 1}-{batch_end}")
        final_results = pd.read_csv(results_file)
        return final_results    


    # lm = setup_databricks()
    # api_processing()


    # -------------------------------------------------------------------------
    # 7. Main loop over row-based batches -  standard way using the server 
    # -------------------------------------------------------------------------
    with tqdm(total=total_rows, desc="Processing Rows") as pbar:
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            # current_batch is a slice of interactions_df by row index
            current_batch = user_df.iloc[batch_start:batch_end]

            # Filter out already processed interactions
            current_batch = current_batch[~current_batch.index.isin(processed_rows)]

            if current_batch.empty:
                pbar.update(batch_end - batch_start)
                continue

            tqdm.write(f"\nProcessing batch {batch_start + 1}-{batch_end} ({len(current_batch)} rows)...")

            # 7a. Select examples
            if last_three:
                examples = last_three
            else:
                examples = initial_examples

            # 7b. Construct the system + example messages
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are the {dataset_name} dataset."
                        "When given a lookup key (e.g., a UserID), you will respond with the exact corresponding Gender,Age,Occupation,Zip-code value from the dataset."
                        "Only respond with the values itself. If the key is unknown, respond with 'Unknown'."
                        "Below are examples of queries and their correct responses:"
                        "Follow this pattern strictly. Let's think step by step.\n\n"
                    )
                }
            ]
            # Add the few-shot examples from either `last_three` or `initial_examples`
            for example in examples:
                messages.extend([
                    {
                        "role": "user",
                        "content": f"{example['UserID']}::"
                    },
                    {
                        "role": "assistant",
                        "content": f"{example['UserID']}::{example['Gender']}::{example['Age']}::{example['Occupation']}::{example['Zip-code']}"
                    },
                ])

            # 7c. Iterate through the rows in the current batch
            for i, row in current_batch.iterrows():
                if i in processed_rows:
                    # If already processed in a previous run, skip
                    pbar.update(1)
                    continue

                # Look up the "UserID" string
                # interaction = user_df.iloc[i]
                interaction = row

                # Append the new user query
                messages.append(
                    {
                        "role": "user",
                        "content": f"{interaction['UserID']}::"
                    }
                )

                # ------------------------------------------------------------------
                # (Optional) If you want to compare to some "real next interaction"
                # or do an error check, define it here. We'll do a dummy check:
                # ------------------------------------------------------------------

                generated_user = query_sglang(sglang_pipeline, messages, config['model_name'])
                # generated_user = query_openai(messages, client)


                if generated_user == 'Unknown':
                    continue

                real_user = interactions_row[i]
                feedback = []
                similarity, _ = adaptive_similarity(generated_user, real_user, feedback)

                similarity = similarity * 100
                similarity_threshold = 70
                if similarity >= similarity_threshold:
                    error_flag = 0
                    logger.info(f"Correct - Generated: '{generated_user}' == Real: '{real_user}' score: {similarity}")
                else:
                    error_flag = 1
                    logger.info(f"Error - Generated: '{generated_user}' <> Real: '{real_user}' score: {similarity}")

                # ------------------------------------------------------------------
                # Save to CSV
                # ------------------------------------------------------------------
                record = {
                    "UserID": interaction['UserID'],
                    "GeneratedUser": generated_user,
                    "RealUser": real_user,
                    "ErrorFlag": error_flag
                }
                try:
                    pd.DataFrame([record]).to_csv(results_file, mode='a', header=False, index=False)
                except Exception as e:
                    logger.error(f"Failed to write record for row {i}: {e}")

                processed_rows.add(i)

                # ------------------------------------------------------------------
                # 7d. Update last_three examples (few-shot) if you want the new row
                #     to become an example.
                # ------------------------------------------------------------------
                # For instance, if there's no error:
                if error_flag == 0:
                    # We'll keep 'InteractionString' as the same.
                    # Or you could parse the model’s response if you want the "next item" specifically.
                    last_three.append(dict(zip(['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], record['RealUser'].split('::'))))
                    if len(last_three) > 3:
                        last_three.pop(0)

                # Add the assistant's response to messages for context
                messages.append(
                    {
                        "role": "assistant",
                        "content": real_user
                    }
                )

                pbar.update(1)
                
                # Proactive rate limiting
                time.sleep(1)

            logger.info(f"Completed batch {batch_start + 1}-{batch_end}")

    logger.info("Processing completed.")
    # Load all results into a DataFrame before returning
    final_results = pd.read_csv(results_file)
    return final_results

# -----------------------------------------------------------------------------
# Title normalization and fuzzy matching helpers.
# -----------------------------------------------------------------------------
def normalize_title(title):
    """
    Normalize the movie title for matching by lowercasing,
    stripping whitespace, and removing punctuation.
    """
    title = title.lower().strip()
    # Remove punctuation (non-alphanumeric characters except whitespace)
    title = re.sub(r'[^\w\s]', '', title)
    return title

def analyze_results(results_df, percentiles=[25, 50, 75, 100]):
    """
    Analyze model coverage (accuracy) at various percentiles and overall.

    Parameters:
        results_df (pd.DataFrame): DataFrame with columns ["MovieID", "GeneratedTitle", "ErrorFlag"].
        percentiles (list): List of percentiles to evaluate coverage.

    Returns:
        list: Analysis report lines.
    """
    total = len(results_df)
    if total == 0:
        return ["No results to analyze."]

    cumsum_errors = results_df["ErrorFlag"].cumsum()
    total_errors = int(cumsum_errors.iloc[-1])
    analysis_report = []

    for percentile in percentiles:
        if percentile < 0 or percentile > 100:
            analysis_report.append(f"Invalid percentile value: {percentile}. Must be between 0 and 100.")
            continue

        # Determine the row index for the given percentile
        cp = int((percentile / 100) * total)
        cp = max(1, cp)  # Ensure at least one row is considered

        err_cp = cumsum_errors.iloc[cp - 1]
        correct_cp = cp - err_cp  # Number of correct predictions
        coverage_cp = correct_cp / cp
        coverage_cp_percentage = round(coverage_cp * 100, 2)
        total_size = cp  # Total number of entries in this percentile

        analysis_report.append(
            f"Coverage at {percentile}th percentile: {coverage_cp_percentage}% | "
            f"Errors: {err_cp} | Correct: {correct_cp} | Total: {total_size}"
        )

    total_coverage = (total - total_errors) / total if total > 0 else 0
    total_coverage_percentage = round(total_coverage * 100, 2)
    analysis_report.append(f"Total Coverage: {total_coverage_percentage}% | Total Errors: {total_errors}")

    return analysis_report

def save_analysis_report(analysis_report, output_file="results/analysis_summary.txt"):
    """
    Save the analysis report lines into a text file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in analysis_report:
            f.write(line + "\n")
    logger.info(f"Analysis saved to {output_file}")

def log_and_save_report(report: list[str], model_name: str, prefix: str = '') -> None:
    """
    Log each line of the report and save it to a file.

    The function prepends an optional prefix to the output filename and logs
    the report content using the module logger.

    Args:
        report (List[str]): List of report lines.
        model_name (str): Model name used to generate the filename.
        prefix (str): Optional prefix for the filename.
    """
    # Log each line in the report.
    for line in report:
        logger.info(line)

    # Build the filename using the model name and prefix.
    filename = f"{prefix}analysis_summary_{model_name}.txt" if prefix else f"analysis_summary_{model_name}.txt"
    # Save the report to the specified file.
    save_analysis_report(report, output_file=filename)


if __name__ == "__main__":

    # Load data Set - da variare a seconda del test
    path = '/Users/claudiosaponaro/Projects/tesi/gepa/users.csv'
    # Use context_size=5 to match the prompt examples (5 items context? Prompt shows 4 items then target)
    # Prompt: "ID: 3049, 3050, 3051, 3052, and target 3053". That's 4 items of context.
    config_path = ('/Users/claudiosaponaro/Projects/tesi/gepa/src/config.yaml')

    try:
        # Open and parse the YAML configuration file.
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")

    # items = load_and_prepare_dataset(path)
    # fetch_movie_name_with_LLM(items, config)

    user_df = load_and_prepare_dataset_user(path)

    small_user_df = user_df[:50]
    user = fetch_user_attribute_with_LLM(user_df, config)

    final_results = pd.read_csv('/Users/claudiosaponaro/Projects/tesi/gepa/meta-llama_Llama-3.2-1B.csv')

    coverage_report = analyze_results(
        final_results,
        percentiles=[1, 10, 20, 25, 50, 75, 90, 100]
    )

    # Log and save the analysis report.
    print(coverage_report)


