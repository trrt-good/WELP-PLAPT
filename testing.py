import torch
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

data=torch.load(r"C:\Users\tatwo\Downloads\encoded_data_lhs")

print(data[0])

# with open('data/output.txt', 'w') as f:
#     for j in range(50):
#         for i in data[j]:
#             f.write(j)
#         f.write("\n")  # If you want a newline after each outer loop iteration
