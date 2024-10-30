import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded data shape: {df.shape}")
    return df

def split_data(df, target_col):
    X = df.drop([target_col, 'date', 'gameweek', 'team', 'is_home'], axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_lasso(X_train, y_train, alpha=0.1):
    model = Lasso(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def create_ensemble(models, X_train, y_train):
    predictions = np.column_stack([model.predict(X_train) for model in models])
    ensemble_model = LinearRegression()
    ensemble_model.fit(predictions, y_train)
    return ensemble_model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return mse, mae

def feature_selection_lasso(X_train, y_train, alpha=0.1):
    lasso = Lasso(alpha=alpha, random_state=42)
    selector = SelectFromModel(lasso, prefit=False)
    selector.fit(X_train, y_train)
    return selector

def main():
    # Load data
    df = load_data('processed_data.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, 'target_xg')
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train individual models
    lr_model = train_linear_regression(X_train_scaled, y_train)
    lasso_model = train_lasso(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train_scaled, y_train)
    gb_model = train_gradient_boosting(X_train_scaled, y_train)
    
    # Create ensemble
    models = [lr_model, lasso_model, rf_model, gb_model]
    ensemble_model = create_ensemble(models, X_train_scaled, y_train)
    
    # Evaluate models
    print("Model Evaluations:")
    for name, model in zip(['Linear Regression', 'Lasso', 'Random Forest', 'Gradient Boosting', 'Ensemble'], models + [ensemble_model]):
        mse, mae = evaluate_model(model, X_test_scaled, y_test)
        print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # Feature selection using Lasso
    lasso_selector = feature_selection_lasso(X_train_scaled, y_train)
    
    # Get selected feature names
    selected_features = X_train.columns[lasso_selector.get_support()]
    print(f"\nSelected features by Lasso:")
    print(selected_features.tolist())
    
    # Train and evaluate model with selected features
    X_train_selected = lasso_selector.transform(X_train_scaled)
    X_test_selected = lasso_selector.transform(X_test_scaled)
    
    lr_model_selected = train_linear_regression(X_train_selected, y_train)
    mse, mae = evaluate_model(lr_model_selected, X_test_selected, y_test)
    print(f"\nLinear Regression with selected features - MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Cross-validation for the selected model
    cv_scores = cross_val_score(lr_model_selected, X_train_selected, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation MSE scores: {-cv_scores}")
    print(f"Average CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

if __name__ == "__main__":
    main()
