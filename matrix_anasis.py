import pandas as pd  # Import pandas for data handling
import numpy as np  # Import NumPy for numerical computations
import logging  # Import logging module
import matplotlib.pyplot as plt
import statistics as stats
import seaborn as sns 


# Configure logging settings
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Read data from an Excel file into a pandas DataFrame
data_content = pd.read_excel("Lab Session Data.xlsx")

# Extract matrix P (columns 1 to 3) and vector Q (column 4)
P = data_content.iloc[:, 1:4].values  # Extract numerical values from DataFrame
Q = data_content.iloc[:, 4].values  # Extract the fourth column as a vector

# Compute and log the number of dimensions (number of columns in P)
dimensions = P.shape[1]
logging.info(f"Number of dimensions in the vector space: {dimensions}")

# Compute and log the total number of vectors (number of rows in P)
vector_count = P.shape[0]
logging.info(f"Total number of vectors in this space: {vector_count}")

# Compute and log the rank of matrix P
matrix_p_rank = np.linalg.matrix_rank(P)
logging.info(f"Rank of Matrix P: {matrix_p_rank}")

# Compute and log the Moore-Penrose Pseudo-Inverse of matrix P
pseudo_inv_p = np.linalg.pinv(P)
logging.info(f"Pseudo-inverse of Matrix P:\n{pseudo_inv_p}\n")

# Compute the cost per product using matrix multiplication
Z = np.dot(pseudo_inv_p, Q)

# Log the computed cost per product
logging.info(f"Computed cost per product:\n{Z}")

# Configure logging settings
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Function to compute the model vector using the pseudo-inverse matrix
def model_vect(A_pinv, R):
    """
    Computes the model vector x using the pseudo-inverse of matrix A and vector R.
    
    Parameters:
    A_pinv (numpy.ndarray): The pseudo-inverse of matrix A.
    R (numpy.ndarray): The vector for which we want to compute the model vector.

    Returns:
    numpy.ndarray: The computed model vector.
    """
    return np.dot(A_pinv, R)  # Perform matrix-vector multiplication

# Compute the model vector using the pseudo-inverse of P and vector Q
x = model_vect(pseudo_inv_p, Q)

# Log the computed model vector
logging.info(f"The model vector x is:\n{x}")

# Configure logging settings
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Load data from an Excel file into a pandas DataFrame
data_content = pd.read_excel("Lab Session Data.xlsx")

# Categorize customers based on their payment amount
# If Payment (Rs) > 200, categorize as 'Rich'; otherwise, categorize as 'Poor'
data_content["Customer_Category"] = data_content["Payment (Rs)"].apply(
    lambda amount: "Rich" if amount > 200 else "Poor"
)

# Extract only relevant columns for classification
classified_data = data_content.loc[:, ["Customer", "Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)", "Customer_Category"]]

# Log the classified data instead of printing
logging.info(f"\n{classified_data}")

# Configure logging settings
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Load stock price data from the specified Excel sheet
stock_data = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

# Extract stock prices as a NumPy array
stock_prices = stock_data["Price"].values

# Compute mean and variance of stock prices
mean_price = stats.mean(stock_prices)
variance_price = stats.variance(stock_prices)

# Log the computed mean and variance
logging.info(f"Mean Stock Price: {mean_price}")
logging.info(f"Variance of Stock Price: {variance_price}")

# Extract stock prices for Wednesdays
wednesday_prices = stock_data[stock_data["Day"] == "Wed"]["Price"].astype(float)

# Compute the mean stock price on Wednesdays
wednesday_mean_price = stats.mean(wednesday_prices)

# Log the computed Wednesday mean price
logging.info(f"Mean Price on Wednesdays: {wednesday_mean_price}")

# Extract stock prices for April month
April_prices = stock_data[stock_data["Month"] == "Apr"]["Price"].astype(float)

# Compute the mean stock price in April
April_mean_price = stats.mean(April_prices)

# Log the computed April mean price
logging.info(f"Mean Price in April: {April_mean_price}")

# Extract percentage change values for Wednesdays and convert to numeric (handling errors)
wednesday_change = pd.to_numeric(stock_data[stock_data["Day"] == "Wed"]["Chg%"], errors="coerce")

# Compute the mean percentage change on Wednesdays
mean_wednesday_change = stats.mean(wednesday_change)

# Log the computed mean percentage change on Wednesdays
logging.info(f"Mean Change% on Wednesdays: {mean_wednesday_change}")

# Convert percentage change values for all days to numeric (handling errors)
change_percentages = pd.to_numeric(stock_data["Chg%"], errors="coerce")

# Compute the probability of profit on Wednesdays (when change% is positive)
profit_wednesday = (wednesday_change > 0).mean()

# Log the probability of profit on Wednesdays
logging.info(f"Probability of Profit on Wednesdays: {profit_wednesday}")

# Compute conditional probability of profit given that it's Wednesday
total_wed_count = len(wednesday_change)
prob_profit_given_wednesday = profit_wednesday if total_wed_count > 0 else 0

# Log the conditional probability
logging.info(f"Conditional Probability of Profit Given It's Wednesday: {prob_profit_given_wednesday}")

# Mapping days of the week to numeric values for visualization
day_mapping = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
stock_data["Day_Numeric"] = stock_data["Day"].map(day_mapping)

# Scatter plot: Stock price change percentage vs. Day of the week
plt.scatter(stock_data["Day_Numeric"], change_percentages)
plt.xlabel("Day of the Week")
plt.ylabel("Change%")
plt.title("Stock Price Change% vs. Day of the Week")
plt.show()

thyroid = pd.read_excel("Lab Session Data.xlsx",sheet_name = "thyroid0387_UCI")

# Configure logging settings
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Function to detect outliers in a given column using the IQR method
def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = data[column].quantile(0.75)  # Third quartile (75th percentile)
    iqr = Q3 - Q1  # Interquartile range
    lb = Q1 - 1.5 * iqr  # Lower bound for outliers
    ub = Q3 + 1.5 * iqr  # Upper bound for outliers

    # Return a boolean mask identifying outlier values
    return (data[column] < lb) | (data[column] > ub)

# Replace '?' values with NaN (missing values)
thyroid.replace('?', np.nan, inplace=True)

# List of numeric test columns
numeric_tests = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']

# Convert numeric test columns to proper numeric format, coercing errors
thyroid[numeric_tests] = thyroid[numeric_tests].apply(pd.to_numeric, errors='coerce')

# Identify columns that contain outliers
outlier_columns = [col for col in numeric_tests if detect_outliers(thyroid, col).sum() > 0]

# Log detected outlier columns
logging.info(f"Columns with outliers detected: {outlier_columns}")

# Handle missing values and outliers
for col in numeric_tests:
    if col in outlier_columns:
        # Replace missing values with the median if column has outliers
        thyroid[col] = thyroid[col].fillna(thyroid[col].median())
        logging.info(f"Missing values in '{col}' replaced with median.")
    else:
        # Replace missing values with the mean if no outliers are present
        thyroid[col] = thyroid[col].fillna(thyroid[col].mean())
        logging.info(f"Missing values in '{col}' replaced with mean.")

# Identify categorical columns
categorical_cols = thyroid.select_dtypes(include=['object']).columns.tolist()

# Handle missing values in categorical columns using mode (most frequent value)
for col in categorical_cols:
    thyroid[col] = thyroid[col].fillna(thyroid[col].mode()[0])
    logging.info(f"Missing values in categorical column '{col}' replaced with mode.")

# Log the dataset cleaning completion
logging.info("The dataset has been successfully cleaned by handling missing values and outliers.")
logging.info(f"Preview of cleaned dataset:\n{thyroid.head()}")

# Save the cleaned dataset to an Excel file using the 'openpyxl' engine
thyroid.to_excel("Imputed_data.xlsx", index=False, engine='openpyxl')
logging.info("Cleaned dataset has been saved as 'Imputed_data.xlsx'.")

# Configure logging settings
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# List of numeric columns to normalize
numeric_columns = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']

# Function to perform Min-Max normalization (scaling between 0 and 1)
def normalize(numeric_cols, df):
    for col in numeric_cols:
        min_val = df[col].min()  # Find the minimum value in the column
        max_val = df[col].max()  # Find the maximum value in the column

        if min_val == max_val:
            logging.warning(f"Column '{col}' has constant values. Skipping normalization.")
            continue  # Skip normalization for constant columns

        # Apply Min-Max normalization formula
        df[col] = (df[col] - min_val) / (max_val - min_val)

        # Log the completion of normalization for each column
        logging.info(f"Normalization applied to column '{col}'.")

# Call the normalization function
normalize(numeric_columns, thyroid)

# Log the completion of normalization
logging.info("Data normalization process completed successfully.")

# Log a preview of the normalized data
logging.info(f"Preview of normalized data:\n{thyroid[numeric_columns].head()}")

# Configure logging settings
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Function to compute **Simple Matching Coefficient (SMC)** and **Jaccard Coefficient (JC)**
def SMC_and_JC(v1, v2):
    # Initialize count variables
    m00, m11, m01, m10 = 0, 0, 0, 0  

    # Iterate through both vectors to compute matches and mismatches
    for i in range(len(v1)):
        if v1[i] == 0 and v2[i] == 0:
            m00 += 1  # Both are 0 (matching)
        elif v1[i] == 1 and v2[i] == 1:
            m11 += 1  # Both are 1 (matching)
        elif v1[i] == 0 and v2[i] == 1:
            m01 += 1  # Mismatch (0 in v1, 1 in v2)
        elif v1[i] == 1 and v2[i] == 0:
            m10 += 1  # Mismatch (1 in v1, 0 in v2)

    # Compute denominators for SMC and JC, handling zero division cases
    denom_smc = m00 + m01 + m10 + m11
    denom_jc = m11 + m01 + m10

    # Compute **Simple Matching Coefficient (SMC)**
    SMC = (m00 + m11) / denom_smc if denom_smc != 0 else 0

    # Compute **Jaccard Coefficient (JC)**
    JC = m11 / denom_jc if denom_jc != 0 else 0

    return SMC, JC  # Return both coefficients

# Binary categorical columns to be converted from 't'/'f' to 1/0
bin_col = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick',
           'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 
           'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych']

# Convert 't' to 1 and 'f' to 0 for binary classification
thyroid[bin_col] = thyroid[bin_col].replace({'t': 1, 'f': 0})

# Extract binary feature vectors for two samples (rows)
vect1 = thyroid.loc[0, bin_col].values  # First sample's binary features
vect2 = thyroid.loc[1, bin_col].values  # Second sample's binary features

# Log the extracted binary vectors
logging.info(f"Binary vector for first sample: {vect1}")
logging.info(f"Binary vector for second sample: {vect2}")

# Compute SMC and JC between the two vectors
SMC, JC = SMC_and_JC(vect1, vect2)

# Log the computed similarity metrics
logging.info(f"Simple Matching Coefficient (SMC): {SMC}")
logging.info(f"Jaccard Coefficient (JC): {JC}")

# Configure logging settings
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Define the numeric columns to be used for cosine similarity calculation
numeric_columns = ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"]

# Function to compute **Cosine Similarity** between two numerical vectors
def cos_sm(v1, v2):
    """
    Computes the Cosine Similarity between two vectors.

    Formula: CS = (A · B) / (||A|| * ||B||)

    - A · B is the dot product of vectors A and B.
    - ||A|| and ||B|| are the magnitudes (norms) of vectors A and B.

    Cosine similarity ranges from **-1 (opposite)** to **1 (identical)**.
    """
    dot_product = np.dot(v1, v2)  # Compute dot product of vectors
    mag_v1 = np.linalg.norm(v1)  # Compute magnitude (norm) of v1
    mag_v2 = np.linalg.norm(v2)  # Compute magnitude (norm) of v2

    # Handle edge case: Avoid division by zero if one of the vectors has zero magnitude
    if mag_v1 == 0 or mag_v2 == 0:
        return 0

    CS = dot_product / (mag_v1 * mag_v2)  # Compute cosine similarity
    return CS  # Return computed similarity score

# Extract numeric feature vectors for two samples (rows)
v1 = thyroid.loc[0, numeric_columns].values  # First sample's numerical features
v2 = thyroid.loc[1, numeric_columns].values  # Second sample's numerical features

# Log extracted numeric vectors
logging.info(f"Numeric vector for first sample: {v1}")
logging.info(f"Numeric vector for second sample: {v2}")

# Compute Cosine Similarity
cosine_similarity = cos_sm(v1, v2)

# Log the computed Cosine Similarity
logging.info(f"Cosine Similarity: {cosine_similarity}")

# Configure logging settings
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Function to plot a heatmap
def heatmap_plot(mtrx, title):
    """
    Plots a heatmap of the given matrix.

    Parameters:
        - mtrx: 2D NumPy array (matrix to visualize)
        - title: Title of the heatmap plot
    """
    plt.figure(figsize=(10, 8))  # Set figure size
    sns.heatmap(mtrx, annot=True, cmap="coolwarm", fmt=".2f")  # Create heatmap with annotations
    plt.title(title)  # Set title
    plt.show()  # Display plot

# List of binary categorical columns in the dataset
binary_column = [
    "on thyroxine", "query on thyroxine", "on antithyroid medication", "sick",
    "pregnant", "thyroid surgery", "I131 treatment", "query hypothyroid", "query hyperthyroid",
    "lithium", "goitre", "tumor", "hypopituitary", "psych"
]

# Convert categorical binary values ('t', 'f') into numeric (1, 0)
thyroid[binary_column] = thyroid[binary_column].replace({'t': 1, 'f': 0})

# Function to compute similarity matrices (Jaccard, SMC, Cosine)
def sim(data):
    """
    Computes similarity matrices for Jaccard Coefficient (JC), Simple Matching Coefficient (SMC),
    and Cosine Similarity (CS).

    Parameters:
        - data: NumPy array (binary feature matrix)

    Returns:
        - smc_mat: SMC similarity matrix
        - jc_mat: Jaccard similarity matrix
        - cos_mat: Cosine similarity matrix
    """
    n = len(data)  # Number of samples
    jc_mat = np.zeros((n, n))  # Initialize Jaccard similarity matrix
    smc_mat = np.zeros((n, n))  # Initialize SMC similarity matrix
    cos_mat = np.zeros((n, n))  # Initialize Cosine similarity matrix

    for i in range(n):
        for j in range(n):
            if i == j:
                # Self-similarity is always 1
                jc_mat[i][j] = 1
                smc_mat[i][j] = 1
                cos_mat[i][j] = 1
            else:
                # Compute Jaccard and SMC similarity
                jc_mat[i][j], smc_mat[i][j] = SMC_and_JC(data[i], data[j])
                # Compute Cosine similarity
                cos_mat[i][j] = cos_sm(data[i], data[j])

    return smc_mat, jc_mat, cos_mat

# Extract first 20 rows for binary similarity computation
data = thyroid.loc[:19, binary_column].values

# Compute similarity matrices
smc_mat, jc_mat, cos_mat = sim(data)

# Log that similarity matrices have been computed
logging.info("Similarity matrices computed successfully.")

# Plot and log heatmaps for different similarity measures
logging.info("Generating heatmap for Simple Matching Coefficient (SMC)...")
heatmap_plot(smc_mat, "SMC Heatmap\n")

logging.info("Generating heatmap for Jaccard Coefficient (JC)...")
heatmap_plot(jc_mat, "JC Heatmap\n")

logging.info("Generating heatmap for Cosine Similarity (CS)...")
heatmap_plot(cos_mat, "CS Heatmap")

