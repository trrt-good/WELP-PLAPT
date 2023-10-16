# Import necessary packages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset

# Load Pre-trained Models and Tokenizers
prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
prot_model = BertModel.from_pretrained("Rostlab/prot_bert")

mol_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
mol_model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# Load Dataset
dataset = load_dataset("jglaser/binding_affinity")

# Preprocess and Tokenize Data
def tokenize_data(data):
    prot_inputs = prot_tokenizer(data['seq'], padding=True, truncation=True, return_tensors="pt")
    mol_inputs = mol_tokenizer(data['smiles_can'], padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(data['affinity']).unsqueeze(1)  # Make sure the shape is (n_samples, 1)
    return prot_inputs, mol_inputs, labels

train_data = dataset['train']
prot_inputs, mol_inputs, labels = tokenize_data(train_data)

# Define Custom Model with Cross Attention
class CrossAttentionModel(nn.Module):
    def __init__(self):
        super(CrossAttentionModel, self).__init__()
        self.prot_encoder = prot_model
        self.mol_encoder = mol_model
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)  # Assume the hidden_size is 768
        self.fc = nn.Linear(768, 1)  # Output layer to predict affinity

    def forward(self, prot_input, mol_input):
        prot_output = self.prot_encoder(**prot_input).last_hidden_state
        mol_output = self.mol_encoder(**mol_input).last_hidden_state
        cross_att_output, _ = self.cross_attention(prot_output, mol_output, mol_output)
        output = self.fc(cross_att_output[:, 0, :])  # Take the [CLS] representation
        return output

# Create DataLoader
batch_size = 32
train_data = TensorDataset(prot_inputs['input_ids'], prot_inputs['attention_mask'],
                           mol_inputs['input_ids'], mol_inputs['attention_mask'],
                           labels)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# Instantiate Model and Loss
model = CrossAttentionModel()
criterion = nn.MSELoss()  # Mean Square Error loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train Model
n_epochs = 10
for epoch in range(n_epochs):
    for batch in train_loader:
        prot_input_ids, prot_attention_mask, mol_input_ids, mol_attention_mask, label = batch
        prot_input = {'input_ids': prot_input_ids, 'attention_mask': prot_attention_mask}
        mol_input = {'input_ids': mol_input_ids, 'attention_mask': mol_attention_mask}
        
        optimizer.zero_grad()
        output = model(prot_input, mol_input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item()}")
