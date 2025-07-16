import pandas as pd
import numpy as np

# Load Excel file and select the worksheet
df = pd.read_excel('Lab Session Data.xlsx', sheet_name='Purchase data')


# Assume last column is the total cost (C) and others are product quantities (A)
A = df.iloc[:, :-1].values  # Features: products bought
C = df.iloc[:, -1].values   # Target: total purchase cost
# Dimensionality = number of columns in A
dimensionality = A.shape[1]

# Number of vectors = number of rows
num_vectors = A.shape[0]

print("Dimensionality of vector space:", dimensionality)
print("Number of vectors in the space:", num_vectors)
rank_A = np.linalg.matrix_rank(A)
print("Rank of Matrix A:", rank_A)
# Compute pseudo-inverse of A
A_pinv = np.linalg.pinv(A)

# Solve for X (cost of each product)
X = A_pinv @ C
print("Estimated cost of each product:", X)


