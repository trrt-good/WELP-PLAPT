
from datasets import load_dataset

dataset = load_dataset('arrow', data_files='data.arrow')


dataset['train'].to_csv('path_to_save.csv')