# test_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def test_models(dataset_path='FNB_Coffee_Roast_Dataset.csv'):
    """
    Loads trained models and evaluates them on the test set.
    """
    print("--- Starting Model Evaluation ---")

    # --- 1. Check for necessary files ---
    required_files = [
        'coffee_quality_model.pkl', 'coffee_anomaly_model.pkl',
        'coffee_scaler.pkl', 'coffee_preprocessors.pkl', dataset_path
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Error: Missing required files: {', '.join(missing_files)}")
        print("Please run the training script first to generate these files.")
        return

    # --- 2. Load Models and Preprocessors ---
    print("Loading saved models and preprocessors...")
    try:
        quality_model = joblib.load('coffee_quality_model.pkl')
        anomaly_model = joblib.load('coffee_anomaly_model.pkl')
        scaler = joblib.load('coffee_scaler.pkl')
        preprocessors = joblib.load('coffee_preprocessors.pkl')
        label_encoders = preprocessors['label_encoders']
        feature_columns = preprocessors['feature_columns']
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model files: {e}")
        return

    # --- 3. Load and Prepare Test Data ---
    print(f"Loading dataset from '{dataset_path}'...")
    df = pd.read_csv(dataset_path)
    df = df.fillna(df.mean(numeric_only=True))

    # Recreate encoded columns to ensure consistency
    for col, le in label_encoders.items():
        df[col + '_encoded'] = le.transform(df[col])

    X = df[feature_columns]
    y_quality = df['overall_quality_score']
    y_anomaly = df['process_anomaly']

    # IMPORTANT: Split the data with the same random_state to get the identical test set
    X_train, X_test, y_quality_train, y_quality_test = train_test_split(X, y_quality, test_size=0.2, random_state=42)
    _, _, y_anomaly_train, y_anomaly_test = train_test_split(X, y_anomaly, test_size=0.2, random_state=42)

    # Scale the test features using the loaded scaler
    X_test_scaled = scaler.transform(X_test)
    print("✅ Test data prepared.")

    # --- 4. Evaluate Quality Model (Regression) ---
    print("\n--- Evaluating Quality Prediction Model ---")
    y_pred_quality = quality_model.predict(X_test_scaled, num_iteration=quality_model.best_iteration)

    rmse = np.sqrt(mean_squared_error(y_quality_test, y_pred_quality))
    mae = mean_absolute_error(y_quality_test, y_pred_quality)
    r2 = r2_score(y_quality_test, y_pred_quality)

    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):   {mae:.4f}")
    print(f"R-squared (R²):              {r2:.4f}")
    print("R² closer to 1.0 is better. RMSE/MAE closer to 0 is better.")


    # --- 5. Evaluate Anomaly Model (Classification) ---
    print("\n--- Evaluating Anomaly Detection Model ---")
    y_pred_anomaly = anomaly_model.predict(X_test_scaled)

    print("Classification Report:")
    # Use target_names if available, otherwise default
    target_names = ['No Anomaly', 'Anomaly']
    print(classification_report(y_anomaly_test, y_pred_anomaly, target_names=target_names))
    print("Key Metrics:")
    print("- Precision: Of all the predictions for a class, how many were correct.")
    print("- Recall: Of all the actual instances of a class, how many did the model find.")
    print("- F1-Score: The harmonic mean of precision and recall.")

    # --- 6. Visualize Results ---
    print("\nGenerating evaluation plots...")
    plt.figure(figsize=(12, 5))

    # Plot 1: Anomaly Detection Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_anomaly_test, y_pred_anomaly)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Anomaly Detection Confusion Matrix')

    # Plot 2: Quality Prediction Residuals
    plt.subplot(1, 2, 2)
    residuals = y_quality_test - y_pred_quality
    plt.scatter(y_pred_quality, residuals, alpha=0.6, edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Quality Score')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Quality Model Residual Plot')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('model_test_evaluation.png')
    plt.show()
    print("✅ Evaluation complete. Plot saved as 'model_test_evaluation.png'.")


if __name__ == "__main__":
    # Ensure you have the dataset in the same directory or provide the correct path
    if not os.path.exists('FNB_Coffee_Roast_Dataset.csv'):
        print("Error: 'FNB_Coffee_Roast_Dataset.csv' not found.")
        print("Please make sure the dataset file is in the same directory as the script.")
    else:
        test_models()
