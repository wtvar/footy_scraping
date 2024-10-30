import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_examine_data(file_path='processed_data.csv'):
    # Load the data
    df = pd.read_csv(file_path)
    print("\n=== Data Overview ===")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation with target
    target_correlations = df[numeric_cols].corr()['target_xg'].sort_values(ascending=False)
    
    print("\n=== Top 10 Feature Correlations with target_xg ===")
    print(target_correlations[:10])
    
    return df, target_correlations

def plot_feature_correlations(correlations, output_file='feature_correlations.png'):
    plt.figure(figsize=(12, 8))
    correlations.plot(kind='bar')
    plt.title('Feature Correlations with target_xg')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def examine_xg_features(df):
    print("\n=== XG-related Features ===")
    xg_cols = [col for col in df.columns if 'xg' in col.lower()]
    print("XG columns found:", xg_cols)
    
    if xg_cols:
        xg_stats = df[xg_cols].describe()
        print("\nXG Features Statistics:")
        print(xg_stats)

def check_feature_selection(df):
    # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in numeric_columns if not col.startswith('target_')]
    
    print("\n=== Feature Selection Analysis ===")
    print(f"Total features available: {len(feature_columns)}")
    print("\nFeatures being used:")
    for col in feature_columns:
        print(f"- {col}")

def analyze_missing_values(df):
    print("\n=== Missing Values Analysis ===")
    missing_vals = df.isnull().sum()
    missing_vals = missing_vals[missing_vals > 0]
    if len(missing_vals) > 0:
        print("\nColumns with missing values:")
        print(missing_vals)
    else:
        print("No missing values found")

def main():
    # Load and examine data
    df, correlations = load_and_examine_data()
    
    # Plot correlations
    plot_feature_correlations(correlations)
    
    # Examine XG features
    examine_xg_features(df)
    
    # Check feature selection
    check_feature_selection(df)
    
    # Analyze missing values
    analyze_missing_values(df)
    
    # Create correlation heatmap for XG-related features
    xg_cols = [col for col in df.columns if 'xg' in col.lower()]
    if len(xg_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[xg_cols].corr(), annot=True, cmap='coolwarm')

if __name__ == "__main__":
    main() 