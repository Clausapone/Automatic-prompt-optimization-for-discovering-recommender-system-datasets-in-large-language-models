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

# =============================================================================
# File for testing bookcrossing memorization using the optimized prompt
# =============================================================================


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

def load_and_prepare_bookmarks(path):
    """
    Loads MovieLens and creates sequential examples.
    """
    print(f"Loading dataset from {path}...")

    try:
        books = pd.read_csv(path, sep=';')
        books = books[["User-ID","ISBN"]]

    except FileNotFoundError:
        print("Warning: 'movies.dat' not found. Creating dummy data.")
        # Dummy data creation omitted for brevity, assuming file exists as per previous context
        return [], [], []

    return books

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
        interactions_row[i] = f"{row['User-ID']}::{row['ISBN']}"

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
            dict(zip(['User-ID', 'ISBN'], record['RealInteraction'].split('::')))
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
    # We can create up to 3 “initial examples” from the first rows:
    initial_examples = []
    max_rows_for_examples = min(3, len(interactions_df))
    for i in range(max_rows_for_examples):
        row = interactions_df.iloc[i]
        init_string = f"{row['User-ID']}::{row['ISBN']}"
        example = {"RowIndex": i, "User-ID":row['User-ID'], "ISBN":row['ISBN'], "InteractionString": init_string}
        initial_examples.append(example)

    # -------------------------------------------------------------------------
    # 6. Prepare row-based batching
    #    Instead of unique user IDs, we simply iterate over the row indices.
    # -------------------------------------------------------------------------
    total_rows = len(interactions_df)

    lm = dspy.LM(
            model='databricks/databricks-meta-llama-3-1-8b-instruct',
            temperature=0.0,  # ✅ ESSENTIAL for memorization testing
            api_base='https://dbc-80ac2274-faac.cloud.databricks.com/serving-endpoints/',
            api_key='dapi43286313f9e8c7796a81aa16957e7200',
            max_tokens=1000
        )
    dspy.configure(lm=lm)
    
    
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
                       """You are an expert on the Book-Crossing dataset, with access to a vast 
                       knowledge base of user reading histories and their corresponding ISBNs. Your task 
                       is to predict the next ISBN a user is likely to have interacted with, given a 
                       provided sequence of user_id -> ISBN mappings.

                       The input format will consist of a context history, which is a sequence of 
                       user_id -> ISBN mappings, and a target user_id. The context history will provide 
                       a list of user_ids and their corresponding ISBNs, and the target user_id will be 
                       the user for which you need to predict the most likely ISBN.

                       To solve this task, you should use the user_id as a key to find the associated 
                       ISBN in your memory. Do not attempt to identify patterns in the context history, 
                       such as sequential user_id ordering, geographic clustering, or rating-based 
                       groupings, as these patterns may not always be present. Instead, rely on your 
                       knowledge of the Book-Crossing dataset to retrieve the correct ISBN for the 
                       target user_id.

                       The Book-Crossing dataset contains a wide range of users from different countries, 
                       age groups, and reading preferences. Each user may be associated with one or 
                       multiple ISBNs reflecting their reading history. Some users have rated books 
                       explicitly (scale 1-10), while others have implicit interactions (rating 0). 
                       ISBNs span genres from literary fiction to self-help, from foreign-language 
                       editions to English bestsellers. Your knowledge base should include this 
                       information to ensure accurate predictions.

                       Some notable examples of correct and incorrect predictions are provided below:

                       * Given the context history
                         user_id: 76726 -> ISBN: 0155061224,
                         user_id: 276727 -> ISBN: 0446520802,
                         user_id: 276729 -> ISBN: 052165615X,
                        user_id: 276733 -> ISBN: 2080674722,
                        user_id: 276736,
                        and the target user_id 276736, the correct prediction is
                        "ISBN: 3257224281".

                        * Given the context history
                        user_id: 276737 -> ISBN: 0600570967,
                        user_id: 276744 -> ISBN: 038550120X,
                        user_id: 276745 -> ISBN: 342310538,
                        user_id: 276746 -> ISBN: 0425115801,
                        user_id: 276747,
                        and the target user_id 276747, the correct prediction is
                        "ISBN: 0449224964".
                        """
                                            )
                }
            ]
            # Add the few-shot examples from either `last_three` or `initial_examples`
            for example in examples:
                messages.extend([
                    {
                        "role": "user",
                        "content": f"{example['User-ID']}::"
                    },
                    {
                        "role": "assistant",
                        "content": f"{example['User-ID']}::{example['ISBN']}"
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
                        "content": f"{interaction['User-ID']}::"
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
                # taking the generated part of the string
                gen_title = generated_user.split('::')[-1].strip()
                real_title = real_interaction.split('::')[-1].strip()
                similarity = compute_similarity(gen_title, real_title)


                similarity_threshold = 70
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
                    last_three.append(dict(zip(['User-ID', 'ISBN'], record['RealInteraction'].split('::'))))
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
    return results_file

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

    # Use context_size=5 to match the prompt examples (5 items context? Prompt shows 4 items then target)
    # Prompt: "ID: 3049, 3050, 3051, 3052, and target 3053". That's 4 items of context.

    path_book = '../datasets/bookcrossing_ratings.csv'

    config_path = '../src/configuration.yaml'

    try:
    # Open and parse the YAML configuration file.
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")

           # Load all results into a DataFrame before returning

    artists = load_and_prepare_bookmarks(path_book)

    # user_interactions = load_and_prepare_dataset_interactions(path_rating)

    # fetch_movie_name_with_LLM(items, config)

    # fetch_user_attribute_with_LLM(user, config)

    final_results = fetch_next_interaction_with_LLM(artists, config)

    print(final_results)
    
    




