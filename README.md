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
