import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import json
from datetime import datetime
import os
import joblib



def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded data shape: {df.shape}")
    return df

def clean_data(df):
    # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

    # Handle numeric columns
    df_numeric = df[numeric_columns]
    
    # Replace infinity with NaN
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    
    # Remove columns with more than 50% missing values
    df_numeric = df_numeric.dropna(axis=1, thresh=len(df_numeric)*0.5)
    
    # Handle non-numeric columns
    df_non_numeric = df[non_numeric_columns]
    
    # Combine cleaned numeric data with non-numeric data
    df_clean = pd.concat([df_numeric, df_non_numeric], axis=1)
    
    print(f"Shape after cleaning: {df_clean.shape}")
    return df_clean

def split_data(df, target_col):
    # Exclude non-numeric columns, the target column, and any other 'target_' columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in numeric_columns if not col.startswith('target_')]
    
    X = df[feature_columns]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def create_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
    ])


def train_linear_regression(X_train, y_train):
    pipeline = create_pipeline()
    model = Pipeline([
        ('preprocessor', pipeline),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)
    return model

def train_lasso(X_train, y_train, alpha=0.1):
    pipeline = create_pipeline()
    model = Pipeline([
        ('preprocessor', pipeline),
        ('regressor', Lasso(alpha=alpha, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    pipeline = create_pipeline()
    model = Pipeline([
        ('preprocessor', pipeline),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    pipeline = create_pipeline()
    model = Pipeline([
        ('preprocessor', pipeline),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model


def create_ensemble(models, X_train, y_train):
    pipeline = create_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    predictions = np.column_stack([model.predict(X_train) for model in models])
    ensemble_model = LinearRegression()
    ensemble_model.fit(predictions, y_train)
    return pipeline, ensemble_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return mse, mae, predictions


def feature_selection_lasso(X_train, y_train, alpha=0.1):
    pipeline = create_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    lasso = Lasso(alpha=alpha, random_state=42)
    selector = SelectFromModel(lasso, prefit=False)
    selector.fit(X_train_processed, y_train)
    return pipeline, selector


def create_feature_importance_plot(model, feature_names, output_file):
    importances = model.named_steps['regressor'].feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_prediction_scatter_plot(y_true, y_pred, output_file):
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    output_dir = 'model_outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    df = load_data('processed_data.csv')
    
    # Clean data
    df_clean = clean_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_clean, 'target_xg')
    
    # Train individual models
    models = {
        'Linear Regression': train_linear_regression(X_train, y_train),
        'Lasso': train_lasso(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
        'Gradient Boosting': train_gradient_boosting(X_train, y_train)
    }
    
    # Save the best performing model (Gradient Boosting)
    best_model = models['Gradient Boosting']
    joblib.dump(best_model, os.path.join(output_dir, 'best_model.pkl'))
    print(f"\nBest model saved as 'best_model.pkl'")

    # Create ensemble
    ensemble_pipeline, ensemble_model = create_ensemble(list(models.values()), X_train, y_train)
    
    # Evaluate models and store results
    results = {}
    for name, model in models.items():
        mse, mae, predictions = evaluate_model(model, X_test, y_test)
        results[name] = {'MSE': mse, 'MAE': mae}
        print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Create prediction scatter plot
        create_prediction_scatter_plot(
            y_test, 
            predictions, 
            os.path.join(output_dir, f'scatter_plot_{name}_{timestamp}.png')
        )
        
        # Create feature importance plot for tree-based models
        if name in ['Random Forest', 'Gradient Boosting']:
            create_feature_importance_plot(
                model, 
                X_train.columns, 
                os.path.join(output_dir, f'feature_importance_{name}_{timestamp}.png')
            )
    
    # Evaluate ensemble model
    ensemble_predictions = np.column_stack([model.predict(X_test) for model in models.values()])
    ensemble_pred = ensemble_model.predict(ensemble_predictions)
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    results['Ensemble'] = {'MSE': ensemble_mse, 'MAE': ensemble_mae}
    print(f"Ensemble - MSE: {ensemble_mse:.4f}, MAE: {ensemble_mae:.4f}")
    
    # Create prediction scatter plot for ensemble
    create_prediction_scatter_plot(
        y_test, 
        ensemble_pred, 
        os.path.join(output_dir, f'scatter_plot_Ensemble_{timestamp}.png')
    )
    
    # Feature selection using Lasso
    lasso_pipeline, lasso_selector = feature_selection_lasso(X_train, y_train)
    
    # Get selected feature names
    X_train_processed = lasso_pipeline.transform(X_train)
    selected_features = X_train.columns[lasso_selector.get_support()].tolist()
    print(f"\nSelected features by Lasso:")
    print(selected_features)
    
    # Train and evaluate model with selected features
    X_train_selected = lasso_selector.transform(X_train_processed)
    X_test_selected = lasso_selector.transform(lasso_pipeline.transform(X_test))
    
    # Train final model with selected features
    final_model = LinearRegression().fit(X_train_selected, y_train)
    final_mse, final_mae, final_predictions = evaluate_model(final_model, X_test_selected, y_test)
    print(f"\nLinear Regression with selected features - MSE: {final_mse:.4f}, MAE: {final_mae:.4f}")
    results['Linear Regression (Selected Features)'] = {'MSE': final_mse, 'MAE': final_mae}

    # Create prediction scatter plot for selected features model
    create_prediction_scatter_plot(
        y_test, 
        final_predictions, 
        os.path.join(output_dir, f'scatter_plot_SelectedFeatures_{timestamp}.png')
    )

    # Cross-validation for the selected model
    cv_scores = cross_val_score(final_model, X_train_selected, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation MSE scores: {-cv_scores}")
    print(f"Average CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save results to JSON file
    results['Selected Features'] = selected_features
    results['Cross Validation'] = {
        'MSE Scores': (-cv_scores).tolist(),
        'Average MSE': float(-cv_scores.mean()),
        'MSE Standard Deviation': float(cv_scores.std() * 2)
    }
    
    with open(os.path.join(output_dir, f'model_results_{timestamp}.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create bar plot of model performances
    plt.figure(figsize=(12, 6))
    model_names = [name for name in results.keys() if name not in ['Selected Features', 'Cross Validation']]
    mse_values = [results[name]['MSE'] for name in model_names]
    
    plt.bar(model_names, mse_values)
    plt.title('Model Comparison - MSE')
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'model_comparison_{timestamp}.png'))
    plt.close()

    return results

if __name__ == "__main__":
    main()

