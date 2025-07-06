import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set visualization style
sns.set(style="whitegrid")

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Print EDA Report to Console
def print_eda_report(df):
    print("="*40)
    print("🔍 BASIC INFORMATION")
    print("="*40)
    print(f"Shape of the dataset: {df.shape}")
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())
    print("\n")
    print("="*40)
    print("📊 DESCRIPTIVE STATISTICS")
    print("="*40)
    print(df.describe(include='all'))
    print("\n")
    print("="*40)
    print("🔗 CORRELATION MATRIX")
    print("="*40)
    print(df.corr(numeric_only=True))
    print("\n")

# Visualizations (optional)
def visualize_data(df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Histograms
    df.hist(figsize=(15, 10), bins=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/histograms.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    # Boxplots for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"{output_dir}/boxplot_{col}.png")
        plt.close()

# Main Function
def main():
    file_path = r"Your Dataset"  # 🔁 Replace with your dataset
    df = load_data(file_path)

    print_eda_report(df)

    print("\n📈 Generating Visualizations...")
    visualize_data(df)

if __name__ == "__main__":
    main()
