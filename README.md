# EDA_Report_Python

🔎 Electric Car Data EDA (Exploratory Data Analysis)
This project provides a simple but powerful Python script for performing Exploratory Data Analysis (EDA) on any structured dataset — with a focus on electric car data. It includes automated summary reporting and visualizations to help understand the structure, quality, and relationships in your data.

📌 Features
📋 Loads and inspects data using pandas

✅ Prints an EDA report to the console:

Shape, data types, missing values, duplicates

Descriptive statistics

Correlation matrix

📈 Generates plots and saves them automatically:

Histograms of all numeric features

Correlation heatmap

Boxplots for each numeric column

Add your dataset

file_path = r"your_dataset.csv"  # Replace with the path to your dataset

📊 Output
All plots are saved in the plots/ directory:

histograms.png — Distribution of all numeric features

correlation_heatmap.png — Heatmap of feature correlations

boxplot_<feature>.png — Boxplots for each numeric column

