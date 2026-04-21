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
import dspy 

logging.basicConfig(level=logging.INFO, format='\n%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class SequentialInteraction(dspy.Signature): 
    '''
    You are a recommender-system sequence completion model.

    Your task is to predict the next item in a user’s interaction sequence based on patterns 
    and similarities observed within the provided dataset {dataset_name}. 
    Change the interaction, has to be not incremental u have to search your own memory, for instance 
    UserID::MovieID --> 1:10, 1: 30, 1:500 until u finish with the first user and go on
    Provide only the requested output and nothing else.

    '''

    user_id = dspy.InputField(
        desc="The specific UserID provided from the Movielens dataset"
    )

    answer = dspy.OutputField(
        desc="The movie id from the Movielens dataset with thi structure -> UserID::MovieID"
    )

def compute_similarity(str_a, str_b):
    """
    Compute similarity (fuzzy match ratio) between two strings using rapidfuzz.

    Parameters:
        str_a (str): First string.
        str_b (str): Second string.

    Returns:
        float: Similarity ratio between 0 and 100.
    """
    return round(SequenceMatcher(None, str_a, str_b).ratio() * 100, 2)

def load_and_prepare_dataset_interactions(path):
    """
    Loads MovieLens and creates sequential examples.
    """
    print(f"Loading dataset from {path}...")

    try:
        movies = pd.read_csv(path)
        movies = movies[["userId","movieId"]]

    except FileNotFoundError:
        print("Warning: 'movies.dat' not found. Creating dummy data.")
        # Dummy data creation omitted for brevity, assuming file exists as per previous context
        return [], [], []

    return movies


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

def fetch_next_interaction_with_LLM(interactions_df, config):


    # -------------------------------------------------------------------------
    # 1. Validate configuration
    # -------------------------------------------------------------------------
    required_keys = ["dataset_name", "model_type", "model_name", "batch_size"]
    if config["model_type"] == "hf":
        required_keys.append("hf_key")
    elif config["model_type"] == "openai":
        required_keys.extend(["azure_endpoint", "azure_openai_key", "api_version", "deployment_name"])
    elif config["model_type"] == "sglang":
        required_keys.append("hf_key")
    elif config["model_type"] == "foundry":
        required_keys.extend(["foundry_model_name", "foundry_endpoint", "foundry_api_key"])
    else:
        raise ValueError("Invalid model_type. Must be one of ['openai','hf','sglang','foundry'].")

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config parameter: {key}")

    dataset_name = config["dataset_name"]
    model_type = config["model_type"]
    batch_size = config["batch_size"]

    # -------------------------------------------------------------------------
    # 2. Create interactions_row dict
    #    Key: row index (int)
    #    Value: "UserID::MovieID"
    # -------------------------------------------------------------------------
    interactions_row = {}
    for i, row in interactions_df.iterrows():
        interactions_row[i] = f"{row['userId']}::{row['movieId']}"

    # -------------------------------------------------------------------------
    # 3. Initialize model/pipeline based on model_type
    #    (Below lines are placeholders; replace with your actual code)
    # -------------------------------------------------------------------------
    
    sglang_pipeline = openai.Client(base_url="http://localhost:7501/v1", api_key="None")
    results_file = f"{config['model_name'].replace('/', '_')}_interaction_results.csv"

    # -------------------------------------------------------------------------
    # 4. Initialize or load existing results
    # -------------------------------------------------------------------------
    if os.path.exists(results_file):
        logger.info(f"Loading existing results from {results_file}.")
        existing_results = pd.read_csv(results_file)
        processed_rows = set(existing_results['RowIndex'].astype(int))
        # Read last few lines for example context
        last_three_record = existing_results.tail(3).to_dict('records')

        # Transform each record into a dictionary with UserID and MovieID
        last_three = [
            dict(zip(['userId', 'movieId'], record['RealInteraction'].split('::')))
            for record in last_three_record
        ]
    else:
        logger.info(f"Creating new results file: {results_file}.")
        existing_results = pd.DataFrame(columns=['RowIndex', 'GeneratedInteraction', 'RealInteraction', 'ErrorFlag'])
        existing_results.to_csv(results_file, index=False)
        processed_rows = set()
        last_three = []  # No previous examples

    # -------------------------------------------------------------------------
    # 5. Define initial examples (few-shot) if no previous examples
    # -------------------------------------------------------------------------
    # To change accordingly to using dataset 
    initial_examples = []
    max_rows_for_examples = min(3, len(interactions_df))
    for i in range(max_rows_for_examples):
        row = interactions_df.iloc[i]
        init_string = f"{row['userId']}::{row['movieId']}"
        example = {"RowIndex": i, "userId":row['userId'], "movieId":row['movieId'], "InteractionString": init_string}
        initial_examples.append(example)

    # -------------------------------------------------------------------------
    # 6. Prepare row-based batching
    #    Instead of unique user IDs, we simply iterate over the row indices.
    # -------------------------------------------------------------------------
    total_rows = len(interactions_df)
    
    # -------------------------------------------------------------------------
    # 7. Initialize DSPy LM using a databricks model, alternatvely is possibile to use a local LLM
    
    lm = dspy.LM(
            model='databricks/databricks-meta-llama-3-1-8b-instruct',
            temperature=0.0,  # ✅ ESSENTIAL for memorization testing
            api_base='https://dbc-80ac2274-faac.cloud.databricks.com/serving-endpoints/',
            api_key='dapi43286313f9e8c7796a81aa16957e7200',
            max_tokens=1000
        )
    dspy.configure(lm=lm)
    

    # Execution using the databricks API model 
    def databricks_execution():

        with tqdm(total=total_rows, desc="Processing Rows") as pbar:
            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                # current_batch is a slice of interactions_df by row index
                current_batch = interactions_df.iloc[batch_start:batch_end]

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

                # 7c. Iterate through the rows in the current batch
                for i, row in current_batch.iterrows():
                    if i in processed_rows:
                        # If already processed in a previous run, skip
                        pbar.update(1)
                        continue


                    # LLM call
                    try:
                        user_id = row['userId']
                        program = dspy.Predict(SequentialInteraction)
                        result = program(user_id=user_id)
                        logger.info(f"Result: {result.answer}")
                        generated_user = result.answer

                        if generated_user == 'Unknown':
                            continue

                    except Exception as e:
                        logger.error(f"Error processing row {i}: {e}")
                        generated_interaction = "Error"

                    # ------------------------------------------------------------------
                    # (Optional) If you want to compare to some "real next interaction"
                    # or do an error check, define it here. We'll do a dummy check:
                    # ------------------------------------------------------------------
                    error_flag = 0  # or 1 if some condition fails

                    real_interaction = interactions_row[i]
                    similarity = compute_similarity(generated_user, real_interaction)

                    similarity_threshold = 80
                    error_flag = 0 if similarity > similarity_threshold else 1
                    if error_flag == 0:
                        logger.info(f"Correct - Generated: '{generated_user}' == Real: '{real_interaction}'")
                    else:
                        logger.info(f"Error - Generated: '{generated_user}' <> Real: '{real_interaction}'")

                    # ------------------------------------------------------------------
                    # Save to CSV
                    # ------------------------------------------------------------------
                    record = {
                        "RowIndex": i,
                        "GeneratedInteraction": generated_user,
                        "RealInteraction": real_interaction,
                        "ErrorFlag": error_flag
                    }
                    try:
                        pd.DataFrame([record]).to_csv(results_file, mode='a', header=False, index=False)
                    except Exception as e:
                        logger.error(f"Failed to write record for row {i}: {e}")

                    processed_rows.add(i)


                    pbar.update(1)

                logger.info(f"Completed batch {batch_start + 1}-{batch_end}")

            logger.info("Processing completed.")
            # Load all results into a DataFrame before returning
            final_results = pd.read_csv(results_file)
            return final_results
    results = databricks_execution()
    return results


    # -------------------------------------------------------------------------
    # 7. Main loop over row-based batches
    # -------------------------------------------------------------------------
    with tqdm(total=total_rows, desc="Processing Rows") as pbar:
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            # current_batch is a slice of interactions_df by row index
            current_batch = interactions_df.iloc[batch_start:batch_end]

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
                        "You are a Recommender Systems. "
                        "Continue user-item interactions list providing the next interaction "
                        f"based on the {dataset_name} dataset. "
                        "When given 'userId, CurrentInteraction', respond with 'userId, NextInteraction'. "
                        "If the next interaction is unknown, respond with 'Unknown'. "
                        #"If there's no next interaction for that user, respond with 'No next interaction'. "
                        "\nBelow are examples of queries and their correct responses:\n\n"
                        "Follow this pattern strictly. Let's think step by step."
                    )
                }
            ]
            # Add the few-shot examples from either `last_three` or `initial_examples`
            for example in examples:
                messages.extend([
                    {
                        "role": "user",
                        "content": f"{example['userId']}::"
                    },
                    {
                        "role": "assistant",
                        "content": f"{example['userId']}::{example['movieId']}"
                    },
                ])

            # 7c. Iterate through the rows in the current batch
            for i, row in current_batch.iterrows():
                if i in processed_rows:
                    # If already processed in a previous run, skip
                    pbar.update(1)
                    continue

                # Look up the "UserID" string
                interaction = interactions_df.iloc[i]

                # Append the new user query
                messages.append(
                    {
                        "role": "user",
                        "content": f"{interaction['userId']}::"
                    }
                )

                # LLM call
                try:
                    generated_user = query_sglang(sglang_pipeline, messages, config['model_name'])

                except Exception as e:
                    logger.error(f"Error processing row {i}: {e}")
                    generated_interaction = "Error"

                # ------------------------------------------------------------------
                # (Optional) If you want to compare to some "real next interaction"
                # or do an error check, define it here. We'll do a dummy check:
                # ------------------------------------------------------------------
                error_flag = 0  # or 1 if some condition fails

                real_interaction = interactions_row[i]
                similarity = compute_similarity(generated_user, real_interaction)

                similarity_threshold = 80
                error_flag = 0 if similarity > similarity_threshold else 1
                if error_flag == 0:
                    logger.info(f"Correct - Generated: '{generated_user}' == Real: '{real_interaction}'")
                else:
                    logger.info(f"Error - Generated: '{generated_user}' <> Real: '{real_interaction}'")

                # ------------------------------------------------------------------
                # Save to CSV
                # ------------------------------------------------------------------
                record = {
                    "RowIndex": i,
                    "GeneratedInteraction": generated_user,
                    "RealInteraction": real_interaction,
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
                    last_three.append(dict(zip(['userId', 'movieId'], record['RealInteraction'].split('::'))))
                    if len(last_three) > 3:
                        last_three.pop(0)

                # Add the assistant's response to messages for context
                messages.append(
                    {
                        "role": "assistant",
                        "content": real_interaction
                    }
                )

                pbar.update(1)

            logger.info(f"Completed batch {batch_start + 1}-{batch_end}")

    logger.info("Processing completed.")
    # Load all results into a DataFrame before returning
    final_results = pd.read_csv(results_file)
    return final_results

def fetch_next_user_interaction_with_LLM(interactions_df, config):
    """
    Given a DataFrame of interactions with columns ["userId","movieId"],
    call a language model to guess (or “continue”) the next user–item interaction,
    defining the batch based on the current user:
      1. Create a dictionary `interactions_row` keyed by row index, with
         values in the format: 'userId::movieId'.
      2. Process the interactions in batches *per user*, e.g. for each user,
         gather all their interactions and move on to the next user.

    Parameters:
        interactions_df (pd.DataFrame): DataFrame with "UserID","MovieID" columns.
        config (dict): Configuration dictionary from YAML.

    Returns:
        pd.DataFrame: DataFrame with columns ["RowIndex", "InteractionString",
                                             "GeneratedOutput", "ErrorFlag"].
    """

    dataset_name = config["dataset_name"]
    model_type = config["model_type"]
    batch_size = config["batch_size"]

    # -------------------------------------------------------------------------
    # 2. Create interactions_row dict
    #    Key: row index (int)
    #    Value: "UserID::MovieID"
    # -------------------------------------------------------------------------
    interactions_row = {}
    for i, row in interactions_df.iterrows():
        interactions_row[i] = f"{row['UserID']}::{row['movieId']}"

    # -------------------------------------------------------------------------
    # 3. Initialize model/pipeline based on model_type
    #    (Below lines are placeholders; replace with your actual code)

    sglang_pipeline = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
    results_file = f"{config['model_name'].replace('/', '_')}_interaction_results.csv"

    # -------------------------------------------------------------------------
    # 4. Initialize or load existing results
    # -------------------------------------------------------------------------
    if os.path.exists(results_file):
        logger.info(f"Loading existing results from {results_file}.")
        existing_results = pd.read_csv(results_file)
        processed_rows = set(existing_results['RowIndex'].astype(int))

        # Example: We won't keep a global last_three, because we will do it per user
        # in the user-based loop. If you still want global examples, adapt as needed.
    else:
        logger.info(f"Creating new results file: {results_file}.")
        existing_results = pd.DataFrame(columns=['RowIndex', 'GeneratedInteraction', 'RealInteraction', 'ErrorFlag'])
        existing_results.to_csv(results_file, index=False)
        processed_rows = set()

    # -------------------------------------------------------------------------
    # 5. Prepare initial examples (few-shot) if you'd like them to be global
    #    E.g. from the first 3 rows overall. (Optional)
    # -------------------------------------------------------------------------
    global_initial_examples = []
    max_rows_for_examples = min(3, len(interactions_df))
    for idx in range(max_rows_for_examples):
        row = interactions_df.iloc[idx]
        init_string = f"{row['userId']}::{row['movieId']}"
        example = {
            "RowIndex": idx,
            "userId": row['userId'],
            "movieId": row['movieId'],
            "InteractionString": init_string
        }
        global_initial_examples.append(example)

    # -------------------------------------------------------------------------
    # 6. Unique users and user-based iteration
    # -------------------------------------------------------------------------
    unique_users = interactions_df['userId'].unique()
    total_users = len(unique_users)

    with tqdm(total=total_users, desc="Processing Users") as user_pbar:
        for user_id in unique_users:
            user_interactions = interactions_df[interactions_df['userId'] == user_id].copy()

            # We will maintain a user-specific "last_three" so that each user has independent context
            # If you need a global context across users, you can move this outside the user loop
            last_three = []

            # If you want to reuse previously processed examples for *this* user, you could load them here:
            # e.g. rows_of_this_user_in_existing_results, parse them into last_three, etc.

            # Now we batch the user's interactions
            num_user_rows = len(user_interactions)
            user_pbar.set_description_str(f"User {user_id} ({num_user_rows} rows)")

            # For large numbers of interactions per user, we still use batch_size
            for batch_start in range(0, num_user_rows, batch_size):
                batch_end = min(batch_start + batch_size, num_user_rows)
                current_batch = user_interactions.iloc[batch_start:batch_end]

                # Filter out already processed interactions
                # (keep them from re-processing if you re-run the script)
                current_batch = current_batch[~current_batch.index.isin(processed_rows)]

                if current_batch.empty:
                    continue

                tqdm.write(f"\nUser {user_id}: processing batch {batch_start + 1}-{batch_end} of {num_user_rows} ...")

                # 6a. Select few-shot examples for this user.
                #     We can mix user-specific 'last_three' with global examples, if desired.
                if last_three:
                    examples = last_three
                else:
                    # Start with global examples or an empty list
                    examples = global_initial_examples

                # 6b. Construct the system + example messages
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a Recommender Systems. "
                            "Continue user-item interactions list providing the next interaction "
                            f"based on the {dataset_name} dataset. "
                            "When given 'UserID::CurrentInteraction', respond with 'UserID::NextInteraction'. "
                            "\nBelow are examples of queries and their correct responses:\n\n"
                            "Follow this pattern strictly. Let's think step by step."
                        )
                    }
                ]
                # Add the few-shot examples
                for example in examples:
                    messages.extend([
                        {
                            "role": "user",
                            "content": f"{example['userId']}::"
                        },
                        {
                            "role": "assistant",
                            "content": f"{example['userId']}::{example['movieId']}"
                        },
                    ])

                # 6c. Iterate through this batch of rows for the current user
                for i, row in current_batch.iterrows():
                    # Safety check in case we re-run:
                    if i in processed_rows:
                        continue

                    # Build user prompt from the row
                    messages.append(
                        {
                            "role": "user",
                            "content": f"{row['userId']}::"
                        }
                    )

                    # LLM call
                    try:
                        if model_type == "openai":
                            output = fetch_with_tenacity(messages, azure_pipeline, config['deployment_name'])
                            if output == 'content_filter_high':
                                generated_interaction = 'Azure Content Filter Error'
                            else:
                                generated_interaction = output.strip()

                        elif model_type == "hf":
                            output = query_hf(messages, hf_pipeline=hf_pipeline)
                            generated_interaction = output[-1]['content'].strip()

                        elif model_type == "sglang":
                            output = query_sglang(sglang_pipeline, messages, config['model_name'])
                            generated_interaction = output[-1]['content'].strip()

                        elif model_type == "foundry":
                            output = query_azure_ai(foundry_pipeline, messages, config['model_name'])
                            generated_interaction = output.strip()

                        else:
                            raise ValueError(f"Unsupported model_type: {model_type}")

                    except Exception as e:
                        logger.error(f"Error processing row {i}: {e}")
                        generated_interaction = "Error"

                    # Optional: some error-check or similarity check
                    real_interaction = interactions_row[i]
                    similarity = compute_similarity(generated_interaction, real_interaction)
                    similarity_threshold = 90
                    error_flag = 0 if similarity > similarity_threshold else 1

                    if error_flag == 0:
                        logger.info(f"Correct - Generated: '{generated_interaction}' == Real: '{real_interaction}'")
                    else:
                        logger.info(f"Error   - Generated: '{generated_interaction}' <> Real: '{real_interaction}'")

                    # ------------------------------------------------------------------
                    # Save to CSV
                    # ------------------------------------------------------------------
                    record = {
                        "RowIndex": i,
                        "GeneratedInteraction": generated_interaction,
                        "RealInteraction": real_interaction,
                        "ErrorFlag": error_flag
                    }
                    try:
                        pd.DataFrame([record]).to_csv(results_file, mode='a', header=False, index=False)
                    except Exception as e:
                        logger.error(f"Failed to write record for row {i}: {e}")

                    processed_rows.add(i)

                    # ------------------------------------------------------------------
                    # 6d. Update user-specific 'last_three' examples if no error
                    # ------------------------------------------------------------------
                    if error_flag == 0:
                        last_three.append(dict(zip(['userId','movieId'], record['RealInteraction'].split('::'))))
                        if len(last_three) > 3:
                            last_three.pop(0)

                    # Add the assistant's response for context (optional)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": real_interaction
                        }
                    )

            user_pbar.update(1)

    logger.info("Processing completed.")
    # Load all results into a DataFrame before returning
    final_results = pd.read_csv(results_file)
    return final_results, results_file

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
    path_item = 'movies.dat'
    # Use context_size=5 to match the prompt examples (5 items context? Prompt shows 4 items then target)
    # Prompt: "ID: 3049, 3050, 3051, 3052, and target 3053". That's 4 items of context.
    path_user = 'users.dat'

    path_rating = '/Users/claudiosaponaro/Projects/tesi/gepa/ratings_100k.csv'
    
    # Load configuration from YAML file
    config_path = '/Users/claudiosaponaro/Projects/tesi/gepa/src/configuration.yaml'

    try:
    # Open and parse the YAML configuration file.
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")

           # Load all results into a DataFrame before returning

    # items = load_and_prepare_dataset(path_item)

    user_interactions = load_and_prepare_dataset_interactions(path_rating)

    # fetch_movie_name_with_LLM(items, config)

    # fetch_user_attribute_with_LLM(user, config)

    final_results = fetch_next_interaction_with_LLM(user_interactions, config)
    
    




