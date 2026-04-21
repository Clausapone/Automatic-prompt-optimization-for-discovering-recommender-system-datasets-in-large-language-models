import pandas as pd
from prova_dipalma_users import analyze_results

final_results = pd.read_csv('/Users/claudiosaponaro/Projects/tesi/gepa/meta-llama_Llama-3.1-8B-ratings-bookcrossing-optimized_interaction_results.csv')


coverage_report = analyze_results(
    final_results,
    percentiles=[1, 10, 20, 25, 50, 75, 90, 100]
)

# Log and save the analysis report.
print(coverage_report)