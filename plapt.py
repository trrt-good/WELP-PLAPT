import torch
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import re
import onnxruntime
import numpy as np

def flatten_list(nested_list):
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            flat_list.extend(flatten_list(element))
        else:
            flat_list.append(element)

    return flat_list

class PredictionModule:
    def __init__(self, model_path="models/affinity_predictor0734-seed2101.onnx"):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Normalization scaling parameters
        self.mean = 6.51286529169358
        self.scale = 1.5614094578916633

    def convert_to_affinity(self, normalized):
        return  {
                    "neg_log10_affinity_M": (normalized * self.scale) + self.mean,
                    "affinity_uM" : (10**6) * (10**(-((normalized * self.scale) + self.mean)))
                }

    def predict(self, batch_data):
        """Run predictions on a batch of data."""
        # Convert each tensor to a numpy array and store in a list
        batch_data = np.array([t.numpy() for t in batch_data])

        # Process each feature in the batch individually and store results
        affinities = []
        for feature in batch_data:
            # Run the model on the single feature
            affinity_normalized = self.session.run(None, {self.input_name: [feature], 'TrainingMode': np.array(False)})[0][0][0]
            # Append the result
            affinities.append(self.convert_to_affinity(affinity_normalized))

        return affinities

class Plapt:
    def __init__(self, prediction_module_path = "models/affinity_predictor0734-seed2101.onnx", caching=True, device='cuda'):
        # Set device for computation
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load protein tokenizer and encoder
        self.prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.prot_encoder = BertModel.from_pretrained("Rostlab/prot_bert").to(self.device)

        # Load molecule tokenizer and encoder
        self.mol_tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.mol_encoder = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(self.device)

        self.caching = caching
        self.cache = {}

        # Load the prediction module ONNX model
        self.prediction_module = PredictionModule(prediction_module_path)

    def set_prediction_module(self, prediction_module_path):
        self.prediction_module = PredictionModule(prediction_module_path)

    @staticmethod
    def preprocess_sequence(seq):
        # Preprocess protein sequence
        return " ".join(re.sub(r"[UZOB]", "X", seq))

    def tokenize(self, prot_seqs, mol_smiles):
        # Tokenize and encode protein sequences
        prot_tokens = self.prot_tokenizer([self.preprocess_sequence(seq) for seq in prot_seqs],
                                            padding=True,
                                            max_length=3200,
                                            truncation=True,
                                            return_tensors='pt')

        # Tokenize and encode molecules
        mol_tokens = self.mol_tokenizer(mol_smiles,
                                            padding=True,
                                            max_length=278,
                                            truncation=True,
                                            return_tensors='pt')
        return prot_tokens, mol_tokens

    # Define the batch functions
    @staticmethod
    def make_batches(iterable, n=1):
        length = len(iterable)
        for ndx in range(0, length, n):
            yield iterable[ndx:min(ndx + n, length)]
    
    def predict_affinity(self, prot_seqs, mol_smiles, batch_size=2):
        input_strs = list(zip(prot_seqs,mol_smiles))
        affinities = []
        for batch in self.make_batches(input_strs, batch_size):
            batch_key = str(batch)  # Convert batch to a string to use as a dictionary key

            if batch_key in self.cache and self.caching:
                # Use cached features if available
                features = self.cache[batch_key]
            else:
                # Tokenize and encode the batch, then cache the results
                prot_tokens, mol_tokens = self.tokenize(*zip(*batch))
                with torch.no_grad():
                    prot_representations = self.prot_encoder(**prot_tokens.to(self.device)).pooler_output.cpu()
                    mol_representations = self.mol_encoder(**mol_tokens.to(self.device)).pooler_output.cpu()

                features = [torch.cat((prot, mol), dim=0) for prot, mol in zip(prot_representations, mol_representations)]

                if self.caching:
                    self.cache[batch_key] = features

            affinities.extend(self.prediction_module.predict(features))

        return affinities
    
    def get_cached_features(self):
        return [tensor.tolist() for tensor in flatten_list(list(self.cache.values()))]

    def clear_cache(self):
        self.cache = {}