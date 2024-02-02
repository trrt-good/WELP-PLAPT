import pandas as pd
from collections import defaultdict
file_path = '/Users/miaanand/Downloads/bindingsitetest.csv'
data = pd.read_csv(file_path)

data.head()

def extract_ligands_and_positions(binding_site_info):
    binding_parts = binding_site_info.split("BINDING")[1:]  # Ignore the first split result
    ligands_positions = defaultdict(list)
    for part in binding_parts:
        details = part.split(";")
        position = details[0].strip()
        ligand = [detail for detail in details if "/ligand=" in detail][0].split("=")[1].strip("\"")
        ligands_positions[ligand].append(position)
    return ligands_positions

data['Ligands_Positions'] = data['Binding site'].apply(extract_ligands_and_positions)

aligned_proteins_df = pd.DataFrame()

for index, row in data.iterrows():
    entry = row['Entry']
    for ligand, positions in row['Ligands_Positions'].items():
        protein_info = f"{entry}: {'; '.join(positions)}"
        if ligand in aligned_proteins_df.columns:
            first_empty_idx = (aligned_proteins_df[ligand].isnull() | (aligned_proteins_df[ligand] == '')).argmax()
            aligned_proteins_df.at[first_empty_idx, ligand] = protein_info
        else:
            aligned_proteins_df = aligned_proteins_df.append({ligand: protein_info}, ignore_index=True)

aligned_proteins_df.fillna('', inplace=True)
aligned_proteins_df.head()
