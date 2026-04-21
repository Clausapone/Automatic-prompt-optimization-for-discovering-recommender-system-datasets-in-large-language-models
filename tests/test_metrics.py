import re

def calculate_age_similarity(age_real, age_pred, feedback, score):
    """
    Similarità basata sulla distanza tra range di età.
    Range adiacenti sono penalizzati meno.
    """
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
    try:
        pos_real = age_order.index(cluster_real)
        pos_pred = age_order.index(cluster_pred)
    except ValueError:
        # Fallback se qualcosa va storto
        feedback.append('Age value not found in ranges')
        score['age_score'] = 0
        return score, feedback

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
        # feedback.append('The occupation is correct') 'riempie troppo il prompt'
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
            # feedback.append('The occupation belong to the right cluster')
            score['occupation_score'] = 0.1
            return score, feedback
        else:
            feedback.append(f"The occupation is incorrect (Real: {occ_real}, Pred: {occ_pred}). Try to identify the correct occupation cluster.")
            score['occupation_score'] = 0
            return score, feedback

    return score, feedback

def calculate_postal_similarity(postal_real, postal_pred, feedback, score):
    """
    Similarità per codici postali con enfasi sulle prime cifre (area geografica).
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

def adaptive_similarity(real_string, pred_string, feedback):
    """
    Metrica che si adatta alla struttura degli ID
    """
    real_parts = real_string.split("::")
    pred_parts = pred_string.split("::")

    # Clean real_parts: remove ID because we want to compare attributes
    # real_string format is defined as: UserID::Gender::Age::Occupation::Zip-code
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
            # feedback.append('The gender is correct')
            score['gender_score'] = 0.30 
        else:
            feedback.append(f"The gender is incorrect (Real: {real_parts[0]}, Pred: {pred_parts[0]})")
            score['gender_score'] = 0

        # Index 1: Age (int)
        try:
            p_age = int(pred_parts[1])
            r_age = int(real_parts[1])
            score, feedback  = calculate_age_similarity(p_age, r_age, feedback, score)
        except (ValueError, IndexError) as e:
            feedback.append(f'Invalid Age format: {pred_parts[1] if len(pred_parts)>1 else "Missing"}. Error: {e}')
            score['age_score'] = 0

        # Index 2: Occupation (int)
        try:
            p_occ = int(pred_parts[2])
            r_occ = int(real_parts[2])
            score, feedback = calculate_occupation_similarity_semantic(p_occ, r_occ, feedback, score)
        except (ValueError, IndexError) as e:
             # Handle "Other service" or similar non-int strings in real or pred
             feedback.append(f'Invalid Occupation format. Real: {real_parts[2] if len(real_parts)>2 else "Missing"}, Pred: {pred_parts[2] if len(pred_parts)>2 else "Missing"}. Error: {e}')
             score['occupation_score'] = 0

        # Index 3: Zip-code (string)
        if len(real_parts) > 3 and len(pred_parts) > 3:
            postal_real = real_parts[3]
            postal_pred = pred_parts[3]
            score, feedback = calculate_postal_similarity(postal_real, postal_pred, feedback, score)
        else:
            feedback.append('Missing Zip-code')
            score['postal_code_score'] = 0
        
        total_score = sum(score.values())
        return total_score, feedback

    except Exception as e:
        # print(f'Errore : {e}') # Suppress print to avoid log spam
        feedback.append(f'General error in similarity calculation: {e}')
        return 0, feedback
