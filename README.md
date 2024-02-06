# **PLAPT: Protein-Ligand Binding Affinity Prediction Using Pretrained Transformer Models**

This is the official code repository for PLAPT, a state-of-the-art 1D sequence-only protein-ligand binding affinity predictor, first introduced [here](https://community.wolfram.com/groups/-/m/t/3094670)

### Abstract
Understanding protein-ligand binding affinity is crucial for drug discovery, enabling the identification of promising drug candidates efficiently. We introduce PLAPT, a novel model leveraging transfer learning from pre-trained transformers like ProtBERT and ChemBERTa to predict binding affinities with high accuracy. Our method processes one-dimensional protein and ligand sequences, leveraging a branching neural network architecture for feature integration and affinity estimation. We demonstrate PLAPT's superior performance through validation on multiple datasets, achieving state-of-the-art results while requiring significantly less computational resources for training compared to existing models. Our findings indicate that PLAPT offers a highly effective and accessible approach for accelerating drug discovery efforts.

![PLAPT Architecture](https://github.com/trrt-good/WELP-PLAPT/blob/main/Diagrams/PLAPT.png)

---
# Usage
---
## Plapt CLI

Plapt CLI is a command-line interface for the Plapt Python package, designed for predicting affinities using sequences and SMILES strings. This tool is user-friendly and offers flexibility in output formats and file handling.

### Prerequisites

Before using Plapt CLI, you need to have the following installed:
- Python (Download and install from [python.org](https://www.python.org/))
- Git (Download and install from [git-scm.com](https://git-scm.com/)) - Alternatively, you can download the repository as a ZIP file.

### Installation

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

### Running the Script

```bash
python plapt_cli.py -s SEQ1 SEQ2 ... -m SMILES1 SMILES2 ... -o OUTPUT_FILE -f FORMAT
```

- `-s`: Followed by one or more sequences.
- `-m`: Followed by one or more SMILES strings.
- `-o`: (Optional) Path to the output file. If omitted, results are printed to the console.
- `-f`: (Optional) Format of the output file (`json` or `csv`). Required if `-o` is used without specifying a file extension.

#### Examples

- To print results to the console:

  ```bash
  python plapt_cli.py -s SEQ1 SEQ2 -m SMILES1 SMILES2
  ```

- To save results to a JSON file:

  ```bash
  python plapt_cli.py -s SEQ1 SEQ2 -m SMILES1 SMILES2 -o results.json
  ```

- To save results to a CSV file:

  ```bash
  python plapt_cli.py -s SEQ1 SEQ2 -m SMILES1 SMILES2 -o results.csv
  ```

- To specify the format explicitly:

  ```bash
  python plapt_cli.py -s SEQ1 SEQ2 -m SMILES1 SMILES2 -o results -f json
  ```

- If `-o` is omitted, results are printed to the console.

---

## Using Plapt Directly in Python

Apart from the command-line interface, Plapt can also be used directly in Python scripts. This allows for more flexibility and integration into larger Python projects or workflows.

### Installation

Ensure you have followed the installation steps mentioned in the earlier section to set up the Plapt environment and dependencies. 

### Basic Usage

To use Plapt in a Python script, you need to import the `Plapt` class and then create an instance of it. You can then call its methods to predict affinities.

#### Importing and Initializing Plapt

``` python
# First, import the Plapt class from the package, making sure you are working in the same directory as the plapt.py file:
from plapt import Plapt

# create an instance of the Plapt class. For basic usage, no initialization parameters are needed:
plapt = Plapt()
```

#### Running Predictions
After initializing the `Plapt` object, you can use it to predict affinities. Here's an example of how to do it:

```python
sequences = ["APTAPSIDMYGSNNL", "PIFLNVLEAIEPGVVC"]
smiles = ["NC(=O)[C@H](CCC(=O)O)", "NC(=[NH2+])c1ccccc1"]

results = plapt.predict_affinity(sequences, smiles)
print(results)
```
output:
```
[{'neg_log10_affinity_M': 4.38891527161495, 'affinity_uM': 40.839905489541835}, {'neg_log10_affinity_M': 4.196127195169673, 'affinity_uM': 63.66090450080189}]
```
The outputted json can subsequently used for other tasks.

### Advanced Usage

Plapt can be initialized with specialized parameters, such as the prediction module used, caching, or the inference device. Example below:
``` python
from plapt import Plapt

# create an instance of the Plapt class with other parameters:
plapt = Plapt(
    prediction_module_path="models/predictionModule.onnx",  # For using a different prediction module. This is set to "models/predictionModule.onnx" by default. 
    caching=True,  # Enable or disable caching. Enabled by default.
    device="cuda"   # Set the computation device ("cuda" for GPU or "cpu" for CPU). If cuda isn't available on your system, it will fallback to "cpu" automatically.
)
```
Each option can be specified seperately (e.g., `plapt = Plapt(caching=False)` if you would like to disable caching. 

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
