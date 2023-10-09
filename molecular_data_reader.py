import pyarrow as pa
import pandas as pd

# Read the Arrow file
with open('path_to_arrow_file.arrow', 'rb') as f:
    reader = pa.ipc.open_file(f)
    table = reader.read_all()

# Convert the table to a pandas DataFrame
df = table.to_pandas()

# Save the DataFrame as a CSV file
df.to_csv('output_file.csv', index=False)
