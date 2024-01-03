# **PLAPT: Protein-Ligand Binding Affinity Prediction Using Pretrained Transformer Models**

This is the official code repository for PLAPT, a state-of-the-art 1D sequence-only protein-ligand binding affinity predictor, first introduced [here](https://community.wolfram.com/groups/-/m/t/3094670)

### Abstract
This study, introduces the Protein Ligand Binding Affinity Prediction Using Pretrained Transformer Models (PLAPT) model, an innovative machine learning approach to predict protein-ligand binding affinities with high accuracy and generalizability, leveraging the wide knowledge of pretrained transformer models. By using ProtBERT and ChemBERTa for encoding protein and ligand sequences, respectively, we trained a two-branch dense neural network that effectively fuses these encodings to estimate binding affinity values. The PLAPT model not only achieves a high Pearson correlation coefficient of ~0.8, but also exhibits no overfitting, a remarkable feat in the context of computational affinity prediction. The robustness of PLAPT, attributed to its generalized transfer learning approach from pre-trained encoders, demonstrates the substantial potential of leveraging extant biochemical knowledge to enhance predictive models in drug discovery.

![PLAPT Architecture](https://github.com/trrt-good/WELP-PLAPT/blob/main/Diagrams/PLAPT.png)

---

# Plapt CLI

Plapt CLI is a command-line interface for the Plapt Python package, designed for predicting affinities using sequences and SMILES strings. This tool is user-friendly and offers flexibility in output formats and file handling.

## Prerequisites

Before using Plapt CLI, you need to have the following installed:
- Python (Download and install from [python.org](https://www.python.org/))
- Git (Download and install from [git-scm.com](https://git-scm.com/)) - Alternatively, you can download the repository as a ZIP file.

## Installation

To install Plapt CLI, you can clone the repository from GitHub:

```bash
git clone https://github.com/trrt-good/WELP-PLAPT.git
cd WELP-PLAPT
```

If you prefer not to use Git, download the ZIP file of the repository and extract it to a desired location.

Once you have the repository on your local machine, install the required dependencies:

```bash
pip install -r requirements.txt
```

(Optional) If you are using a virtual environment, activate it before installing the dependencies:

```bash
source /path/to/your/venv/bin/activate
```

## Usage

Navigate to the directory where you cloned or extracted the Plapt CLI repository. The `plapt_cli.py` script is your main entry point.

### Basic Usage

Run the script with the following command:

```bash
python plapt_cli.py -s [SEQUENCES] -m [SMILES]
```

Replace `[SEQUENCES]` and `[SMILES]` with your lists of sequences and SMILES strings, respectively. **Do not use brackets**.

### Advanced Usage

#### Output to a File

To save results to a file, use the `-o` option:

```bash
python plapt_cli.py -s SEQ1 SEQ2 -m SMILES1 SMILES2 -o results.json
```

#### Specify Output Format

To define the format of the output file (JSON or CSV), use the `-f` option:

```bash
python plapt_cli.py -s SEQ1 SEQ2 -m SMILES1 SMILES2 -o results -f csv
```
---

#### Data Preparation and Encoding
We source protein-ligand pairs and their corresponding affinity values from an open-source binding affinity dataset on hugginface, [binding_affinity](https://huggingface.co/datasets/jglaser/binding_affinity). We then used ProtBERT and ChemBERTa for encoding proteins and ligands respectively, giving us high quality vector-space representations. The encoding process is detailed in the `encoding.ipynb` notebook. The dataset, already encoded, is available on our [Google Drive](https://drive.google.com/drive/folders/1e-ujgHx5bW0JKxSZY5u34As77o4-IIFs?usp=sharing) for ease of access and use.

#### Importing Encoders and Running the Notebook
For users to import the encoders and run the Wolfram notebook (`WL Notebooks/FinalEssay.nb`), we provide the `encoders_to_onnx.ipynb` notebook. This ensures that users can replicate our encoding process and utilize the full capabilities of PLAPT.

### Results

PLAPT achieved impressive results, demonstrating both high accuracy and state-of-the-art generalization in protein-ligand binding affinity prediction. Detailed analysis can be found in our paper. Key metrics include:

| Metric | Test Data | Train Data |
| ------ | --------- | ---------- |
| R (Pearson Correlation) | 0.800988 | 0.798657 |
| MSE (Mean Squared Error) | 0.978599 | 0.967477 |
| RMSE (Root Mean Squared Error) | 0.989241 | 0.983604 |
| MAE (Mean Absolute Error) | 0.864218 | 0.861717 |

# **PLAPT: Protein-Ligand Binding Affinity Prediction Using Pretrained Transformer Models**

This is the official code repository for [PLAPT](https://github.com/trrt-good/WELP-PLAPT/blob/main/PLAPT.pdf)

### Abstract
This study, introduces the Protein Ligand Binding Affinity Prediction Using Pretrained Transformer Models (PLAPT) model, an innovative machine learning approach to predict protein-ligand binding affinities with high accuracy and generalizability, leveraging the wide knowledge of pretrained transformer models. By using ProtBERT and ChemBERTa for encoding protein and ligand sequences, respectively, we trained a two-branch dense neural network that effectively fuses these encodings to estimate binding affinity values. The PLAPT model not only achieves a high Pearson correlation coefficient of ~0.8, but also exhibits no overfitting, a remarkable feat in the context of computational affinity prediction. The robustness of PLAPT, attributed to its generalized transfer learning approach from pre-trained encoders, demonstrates the substantial potential of leveraging extant biochemical knowledge to enhance predictive models in drug discovery.

![PLAPT Results Graph](https://github.com/trrt-good/WELP-PLAPT/blob/main/Diagrams/PLAPT.png)

#### Data Preparation and Encoding
We sourcing protein-ligand pairs and their corresponding affinity values from an open-source binding affinity dataset on hugginface, [binding_affinity](https://huggingface.co/datasets/jglaser/binding_affinity). We then used ProtBERT and ChemBERTa for encoding proteins and ligands respectively, giving us high quality vector-space representations. The encoding process is detailed in the `encoding.ipynb` notebook. The dataset, already encoded, is available on our [Google Drive](https://drive.google.com/drive/folders/1e-ujgHx5bW0JKxSZY5u34As77o4-IIFs?usp=sharing) for ease of access and use.

#### Importing Encoders and Running the Notebook
For users to import the encoders and run the Wolfram notebook (`WL Notebooks/FinalEssay.nb`), we provide the `encoders_to_onnx.ipynb` notebook. This ensures that users can replicate our encoding process and utilize the full capabilities of PLAPT.

### Results

PLAPT achieved impressive results, demonstrating both high accuracy and state-of-the-art generalization in protein-ligand binding affinity prediction. Detailed analysis can be found in our paper. Key metrics include:

| Metric | Test Data | Train Data |
| ------ | --------- | ---------- |
| R (Pearson Correlation) | 0.800988 | 0.798657 |
| MSE (Mean Squared Error) | 0.978599 | 0.967477 |
| RMSE (Root Mean Squared Error) | 0.989241 | 0.983604 |
| MAE (Mean Absolute Error) | 0.864218 | 0.861717 |

![PLAPT Results Graph](https://github.com/trrt-good/WELP-PLAPT/blob/main/Diagrams/Graphs.png)
<p align = "center">
<img width="100%" alt="Screenshot 2023-12-28 at 7 51 22 AM" src="https://github.com/trrt-good/WELP-PLAPT/assets/25653940/f3051e72-a669-4425-bb09-f82ac52b14a8">
<img width="100%" alt="Screenshot 2023-12-28 at 7 51 30 AM" src="https://github.com/trrt-good/WELP-PLAPT/assets/25653940/8367503c-d0de-41c2-9254-aec4570b93d6">
</p>
