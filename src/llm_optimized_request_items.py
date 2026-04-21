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
from openai import AzureOpenAI


def load_and_prepare_dataset(path):
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

def compute_similarity_item(str_a, str_b):
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

logging.basicConfig(level=logging.INFO, format='\n%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def compute_similarity_user(str_a, str_b):
    """
    Compute similarity (fuzzy match ratio) between two strings using rapidfuzz.

    Parameters:
        str_a (str): First string.
        str_b (str): Second string.

    Returns:
        float: Similarity ratio between 0 and 100.
    """
    '''result = adaptive_similarity(str_a, str_b)
    return round(result * 100, 2)'''

def load_and_prepare_user(path):
    """
    Carica il dataset users.dat di MovieLens.
    
    Formato: UserID::Gender::Age::Occupation::Zip-code
    
    Parameters:
        path (str): Percorso del file users.dat
        
    Returns:
        pd.DataFrame: DataFrame con le colonne degli utenti
    """
    print(f"Caricando il dataset users da {path}...")
    
    try:
        users = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
            encoding="latin-1"
        )
        
        print(f"Dataset caricato con successo! Totale utenti: {len(users)}")
        return users
        
    except FileNotFoundError:
        print(f"Errore: file '{path}' non trovato!")
        return None

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


    """
    Query the OpenAI with OpenAI compatible API

    Parameters:
        azure_pipeline (object): The pipeline used to interact with the OpenAI compatible API.
        messages (list): A list of messages as dictionaries (e.g., {"role": "user", "content": "Your message"}).
        model (str): The model name (e.g., "meta-llama/Llama-3.1-405B-Instruct-FP8").

    Returns:
        str: Generated content from the model.
    """
    # Generate the completion
    completion = azure_pipeline.chat.completions.create(
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

def query_azure_ai(foundry_pipeline, messages, model):
    """
    Query the Foundry with OpenAI compatible API

    Parameters:
        foundry_pipeline (object): The pipeline used to interact with the Foundry compatible OpenAI API.
        messages (list): A list of messages as dictionaries (e.g., {"role": "user", "content": "Your message"}).
        model (str): The model name (e.g., "meta-llama/Llama-3.1-405B-Instruct-FP8").

    Returns:
        str: Generated content from the model.
    """
    payload = {
        'messages': messages,
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "seed": 42
    }

    # Generate the completion
    completion = foundry_pipeline.complete(payload)

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
    id_to_title = dict(zip(items_df['movieId'], items_df['title']))
    batch_size = 50
    sglang_pipeline = openai.Client(base_url="http://localhost:7501/v1", api_key="None")
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
        {"MovieID": str(items_df.iloc[0]['movieId']), "RealTitle": items_df.iloc[0]['title']},
        {"MovieID": str(items_df.iloc[1]['movieId']), "RealTitle": items_df.iloc[1]['title']},
        {"MovieID": str(items_df.iloc[2]['movieId']), "RealTitle": items_df.iloc[2]['title']},
    ]

    # Prepare all movie IDs as strings
    movie_ids = items_df["movieId"].astype(str).tolist()
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
                        '''
                        f"You are the {dataset_name} dataset. "
                        "When given a lookup key (e.g., a MovieID), you will respond with the exact corresponding value from the dataset. "
                        "Only respond with the value itself. If the key is unknown, respond with 'Unknown'. "
                        "Below are examples of queries and their correct responses:\n\n"
                        "Follow this pattern strictly. Let's think step by step."
                        '''
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
                # output = query_azure_ai(foundry_pipeline, messages, config['model_name'])
                if output == 'Unknown':
                    continue

                generated_title =  re.sub(r"\s*\(\d{4}\)$", "", re.sub(r"^\d+::\s*", "", output))
                real_title = id_to_title.get(int(movie_id_str), "Unknown")
                similarity = compute_similarity_item(generated_title, real_title)

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
    return results_file

def fetch_user_attribute_with_LLM(user_df, config):
    """
    Given a DataFrame of interactions with columns ["UserID","MovieID"],
    call a language model to guess (or “continue”) the next user–item interaction,
    using a row-based approach:
      1. Create a dictionary `interactions_row` keyed by row index, with
         values in the format: 'UserID::MovieID'.
      2. Batch-process rows (instead of unique user IDs).

    The system prompt is set up for a Recommender System:
      "You are a Recommender Systems. Continue user-item interactions list
       providing the next interaction based on the MovieLens1M dataset.
       When given 'UserID, CurrentInteraction', respond with 'UserID, NextInteraction'.
       If there's no next interaction for that user, respond with 'No next interaction'."

    Parameters:
        user_df (pd.DataFrame): DataFrame with "UserID","MovieID" columns.
        config (dict): Configuration dictionary from YAML.

    Returns:
        pd.DataFrame: DataFrame with columns ["RowIndex", "InteractionString",
                                             "GeneratedOutput", "ErrorFlag"].
    """



    '''dataset_name = config["dataset_name"]
    model_type = config["model_type"]
    batch_size = config["batch_size"]

    # aggiugnere sglang 

    interactions_row = {}
    for i, row in user_df.iterrows():
        interactions_row[i] = f"{row['UserID']}::{row['Gender']}::{row['Age']}::{row['Occupation']}::{row['Zip-code']}"

    sglang_pipeline = openai.Client(base_url="http://localhost:7501/v1", api_key="none")
    results_file = f"{config['model_name'].replace('/', '_')}_users_results.csv"
    
    processed_rows = set()
    last_three = []  # No previous examples

    # -------------------------------------------------------------------------
    # 5. Define initial examples (few-shot) if no previous examples
    # -------------------------------------------------------------------------
    # We can create up to 3 “initial examples” from the first rows:
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

    # -------------------------------------------------------------------------
    # 7. Main loop over row-based batches
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
                interaction = user_df.iloc[i]

                # Append the new user query
                messages.append(
                    {
                        "role": "user",
                        "content": f"{interaction['UserID']}::"
                    }
                )

                output = query_sglang(sglang_pipeline, messages, config['model_name'])
                if output == 'Unknown':
                    continue


                # l'output è la interaction string quindi in GEPA devo allenare il modello con questo task

                # ------------------------------------------------------------------
                # (Optional) If you want to compare to some "real next interaction"
                # or do an error check, define it here. We'll do a dummy check:
                # ------------------------------------------------------------------
                error_flag = 0  # or 1 if some condition fails

                real_user = interactions_row[i]
                similarity = compute_similarity_user(output, real_user)

                similarity_threshold = 7
                error_flag = 0 if similarity >= similarity_threshold else 1
                if error_flag == 0:
                    logger.info(f"Correct - Generated: '{output}' == Real: '{real_user}'")
                else:
                    logger.info(f"Error - Generated: '{output}' <> Real: '{real_user}'")

                # ------------------------------------------------------------------
                # Save to CSV
                # ------------------------------------------------------------------
                record = {
                    "RowIndex": i,
                    "GeneratedInteraction": output,
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

            logger.info(f"Completed batch {batch_start + 1}-{batch_end}")

    logger.info("Processing completed.")
    # Load all results into a DataFrame before returning
    final_results = pd.read_csv(results_file)
    return final_results'''



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

    try :
        cumsum_errors = results_df["ErrorFlag"].cumsum()
    except KeyError:
        cumsum_errors = results_df["ErrorFlag"].cumsum()
    print(cumsum_errors)
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

    # Load Test Set
    # path_item = 'movies.dat'
    path_item = 'movies_100k.csv'
    # Use context_size=5 to match the prompt examples (5 items context? Prompt shows 4 items then target)
    # Prompt: "ID: 3049, 3050, 3051, 3052, and target 3053". That's 4 items of context.
    path_user = 'users.dat'
    # path_item = 'movies.dat'
    config_path = ('/Users/claudiosaponaro/Projects/tesi/gepa/src/configuration.yaml')

    try:
    # Open and parse the YAML configuration file.
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")

           # Load all results into a DataFrame before returning

    # items = load_and_prepare_dataset(path_item)

    # user = load_and_prepare_user(path_user)

    items = pd.read_csv(path_item, encoding='latin-1')

    result_file = fetch_movie_name_with_LLM(items, config)
    print(result_file)

    # fetch_user_attribute_with_LLM(user, config)
    

    




