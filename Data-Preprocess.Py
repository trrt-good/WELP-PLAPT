import pandas as pd
import re
from collections import defaultdict

def load_and_process_csv(file_path):
    df = pd.read_csv(file_path)

    def parse_binding_sites(binding_site_info):
        pattern = r'BINDING (\d+).*?ChEBI:CHEBI:(\d+)'
        bindings = re.findall(pattern, binding_site_info)
        grouped_bindings = defaultdict(list)
        for position, chebi_id in bindings:
            grouped_bindings[chebi_id].append(int(position))
        return grouped_bindings
    binding_info = df['Binding site'].apply(parse_binding_sites)

    aligned_data = []
    for entry, bindings in zip(df['Entry'], binding_info):
        for chebi_id, positions in bindings.items():
            aligned_data.append([entry, chebi_id, positions])

    aligned_df = pd.DataFrame(aligned_data, columns=['seq', 'chebi_id', 'binding_sites'])

    return aligned_df

def save_aligned_data(input_file_path, output_file_path):
    aligned_df = load_and_process_csv(input_file_path)
    aligned_df.to_csv(output_file_path, index=False)

file_path = 'input.csv'
aligned_csv_path = 'output.csv'

save_aligned_data(file_path, aligned_csv_path)
