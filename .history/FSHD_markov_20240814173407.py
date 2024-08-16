import numpy as np
from scipy.stats import dirichlet, chisquare
import matplotlib.pyplot as plt
import random
# Data retrieval
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

# # Define the sample file mappings (local paths)
# samples = {
#     "FSHD1.1": "/Users/chris/iGEM_modeling/scRNAseqData/GSM3487556_FSHD1.1.txt",
#     "FSHD1.2": "/Users/chris/iGEM_modeling/scRNAseqData/GSM3487557_FSHD1.2.txt"
#     }

# # Initialize an empty dictionary to store the AnnData objects
# adatas = {}

# # Read and process data for each sample
# for sample_id, filepath in samples.items():
#     try:
#         # Read the text file into a DataFrame 
#         sample_data = pd.read_csv(filepath, sep="\t", index_col=0)
        
#         # Convert the DataFrame to a sparse matrix
#         sample_matrix = csr_matrix(sample_data.values)
#         # Create an AnnData object
#         sample_adata = ad.AnnData(sample_matrix)

#         # Set the observation (cell) names and variable (gene) names
#         sample_adata.var_names_make_unique()  # Ensure gene names are unique
#         sample_adata.obs_names = sample_data.index.tolist()
#         sample_adata.var_names = sample_data.columns.tolist()
        
#         # Store the AnnData object in the dictionary
#         adatas[sample_id] = sample_adata
        
#         print(f"Successfully read data for {sample_id}")
#     except Exception as e:
#         print(f"Failed to read data for {sample_id}: {e}")

# # The adatas dictionary now contains AnnData objects for each sample
# for sample_id, adata in adatas.items():
#     print(f"Data for {sample_id}:")
#     print(adata)  # Print a summary of each AnnData object
#     print(adata.X)  # Print the data matrix

# # Optional: Print the observation and variable names for the first sample
# if adatas:
#     sample_id = list(adatas.keys())[0]
#     adata = adatas[sample_id]
#     print("Observation (cell) names:", adata.obs_names[:10])
#     print("Variable (gene) names:", adata.var_names[:10])

# adata.layers["log_transformed"] = np.log1p(adata.X)
# adata

# adata.to_df(layer="log_transformed")
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
autonsitions
auto
auto
auto
auto
auto
auto
auto
auto
autoities[non_zero_indices], 1/24)
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
autotribution, iterations=10000):
auto
auto
auto
auto
auto
auto
autoilities
auto
auto
auto
auto
auto
auto
auto
auto
autoribution, 72)  # Simulate over 72 hours
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
autoe in predicted_distribution)
auto
auto
auto
autote_distribution, observed_state_distribution_3days)
autoon, 72)  # Simulate over 72 hours
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
autostate]}")
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto
auto