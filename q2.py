import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats
from pathlib import Path

# Configure logging format and level
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Validate file existence before processing
def validate_excel_file(path):
    if not Path(path).is_file():
        logging.error(f"File missing at: {path}")
        raise FileNotFoundError(f"Missing Excel file at: {path}")
    logging.info(f"File confirmed: {path}")

# Section 1: Perform linear algebra analysis on purchases
def evaluate_shopping_data(path):
    df = pd.read_excel(path, sheet_name="Purchase data")
    inputs = df.iloc[:, 1:4].values
    outputs = df.iloc[:, 4].values

    cols = inputs.shape[1]
    rows = inputs.shape[0]
    rank = np.linalg.matrix_rank(inputs)

    pinv = np.linalg.pinv(inputs)
    item_costs = np.dot(pinv, outputs)

    logging.info("=== Shopping Data Matrix Evaluation ===")
    logging.info(f"Input matrix columns: {cols}")
    logging.info(f"Total records: {rows}")
    logging.info(f"Matrix rank: {rank}")
    logging.info(f"Pseudo-inverse matrix:\n{pinv}")
    logging.info(f"Estimated item costs:\n{item_costs}")

    return inputs, outputs, pinv, item_costs

# Section 2: Segment customers into categories
def tag_customer_segments(path):
    df = pd.read_excel(path, sheet_name="Purchase data")
    df["Segment"] = df["Payment (Rs)"].apply(lambda x: "Rich" if x > 200 else "Poor")
    labeled = df[["Customer", "Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)", "Segment"]]

    logging.info("=== Labeled Customer Categories ===")
    logging.info(f"\n{labeled}")
    return labeled

# Section 3: Stock trend summarization
def summarize_stock_data(path):
    df = pd.read_excel(path, sheet_name="IRCTC Stock Price")
    price_list = df["Price"].values

    avg_price = stats.mean(price_list)
    var_price = stats.variance(price_list)

    wed_only = df[df["Day"] == "Wed"]["Price"].astype(float)
    wed_avg = stats.mean(wed_only) if not wed_only.empty else 0

    apr_data = df[df["Month"] == "Apr"]["Price"].astype(float)
    apr_avg = stats.mean(apr_data) if not apr_data.empty else 0

    wed_chg = pd.to_numeric(df[df["Day"] == "Wed"]["Chg%"], errors="coerce")
    avg_wed_chg = stats.mean(wed_chg.dropna()) if not wed_chg.dropna().empty else 0
    prob_profit = (wed_chg > 0).mean() if not wed_chg.dropna().empty else 0

    df["Day_Num"] = df["Day"].map({"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7})
    chg_vals = pd.to_numeric(df["Chg%"], errors="coerce")
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Day_Num"], chg_vals, alpha=0.6)
    plt.xlabel("Weekday Index")
    plt.ylabel("Change in Price (%)")
    plt.title("Weekly Price Movement")
    plt.grid(True)
    plt.show()

    logging.info("=== IRCTC Stock Summary ===")
    logging.info(f"Mean price: {avg_price:.2f}")
    logging.info(f"Price variance: {var_price:.2f}")
    logging.info(f"Wednesday Avg: {wed_avg:.2f}")
    logging.info(f"April Avg: {apr_avg:.2f}")
    logging.info(f"Wed % Change Avg: {avg_wed_chg:.2f}")
    logging.info(f"Profit Likelihood on Wednesdays: {prob_profit:.2f}")

    return avg_price, var_price, wed_avg, apr_avg, prob_profit

# Section 4: Clean and scale thyroid medical data
def clean_thyroid_info(path):
    df = pd.read_excel(path, sheet_name="thyroid0387_UCI")
    df.replace("?", np.nan, inplace=True)

    features = ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")

    def detect_outliers(col):
        q1, q3 = col.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return (col < lower) | (col > upper)

    flagged = [col for col in features if detect_outliers(df[col]).sum() > 0]
    logging.info(f"Outlier-containing columns: {flagged}")

    for col in features:
        if col in flagged:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    cats = df.select_dtypes(include="object").columns
    for cat in cats:
        df[cat].fillna(df[cat].mode()[0], inplace=True)

    for col in features:
        min_val, max_val = df[col].min(), df[col].max()
        if min_val != max_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            logging.warning(f"No scaling for {col} (constant)")

    df.to_excel("Imputed_data.xlsx", index=False, engine="openpyxl")
    logging.info("Saved cleaned thyroid dataset to Imputed_data.xlsx")

    return df, features

# Section 5: Compute similarity matrices
def calculate_pairwise_similarity(df, bin_fields, num_fields):
    df[bin_fields] = df[bin_fields].replace({"t": 1, "f": 0})

    def smc_jaccard(v1, v2):
        both_zero = sum((v1 == 0) & (v2 == 0))
        both_one = sum((v1 == 1) & (v2 == 1))
        one_zero = sum((v1 == 1) & (v2 == 0))
        zero_one = sum((v1 == 0) & (v2 == 1))
        smc = (both_one + both_zero) / (both_one + both_zero + one_zero + zero_one)
        jc = both_one / (both_one + one_zero + zero_one) if (both_one + one_zero + zero_one) > 0 else 0
        return smc, jc

    def cosine_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) if np.linalg.norm(v1) and np.linalg.norm(v2) else 0

    subset_bin = df.loc[:19, bin_fields].values
    subset_num = df.loc[:19, num_fields].values
    size = len(subset_bin)

    smc_mat = np.zeros((size, size))
    jc_mat = np.zeros((size, size))
    cos_mat = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i == j:
                smc_mat[i, j] = jc_mat[i, j] = cos_mat[i, j] = 1
            else:
                smc_mat[i, j], jc_mat[i, j] = smc_jaccard(subset_bin[i], subset_bin[j])
                cos_mat[i, j] = cosine_sim(subset_num[i], subset_num[j])

    def render_heatmap(matrix, heading):
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, cmap="coolwarm", annot=True, fmt=".2f")
        plt.title(heading)
        plt.show()

    logging.info("Generating similarity heatmaps...")
    render_heatmap(smc_mat, "SMC Heatmap")
    render_heatmap(jc_mat, "Jaccard Heatmap")
    render_heatmap(cos_mat, "Cosine Heatmap")

    logging.info(f"Example SMC (0 vs 1): {smc_mat[0,1]:.2f}")
    logging.info(f"Example Jaccard (0 vs 1): {jc_mat[0,1]:.2f}")
    logging.info(f"Example Cosine (0 vs 1): {cos_mat[0,1]:.2f}")

    return smc_mat, jc_mat, cos_mat

# Main driver
def driver():
    dataset_path = r"C:\Users\akshath\OneDrive\Desktop\mlass2\Lab Session Data (2).xlsx"
    validate_excel_file(dataset_path)

    logging.info("Running matrix decomposition for purchase entries...")
    evaluate_shopping_data(dataset_path)

    logging.info("Categorizing customers...")
    tag_customer_segments(dataset_path)

    logging.info("Parsing stock prices...")
    summarize_stock_data(dataset_path)

    logging.info("Cleaning thyroid records...")
    thyroid_df, num_cols = clean_thyroid_info(dataset_path)

    binary_attributes = [
        "on thyroxine", "query on thyroxine", "on antithyroid medication", "sick",
        "pregnant", "thyroid surgery", "I131 treatment", "query hypothyroid", 
        "query hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych"
    ]

    logging.info("Computing pairwise metrics...")
    calculate_pairwise_similarity(thyroid_df, binary_attributes, num_cols)

    logging.info("Finished all tasks successfully.")

# Run the script
if __name__ == "__main__":
    driver()
