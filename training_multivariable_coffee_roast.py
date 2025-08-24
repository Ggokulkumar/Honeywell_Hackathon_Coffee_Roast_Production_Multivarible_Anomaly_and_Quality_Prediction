import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           classification_report, confusion_matrix, roc_auc_score,
                           precision_recall_curve, auc, f1_score, accuracy_score)
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class EnhancedCoffeeRoastPredictor:
    def __init__(self):
        self.quality_model = None
        self.anomaly_model = None
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.feature_selector = None
        self.pca = None
        self.use_pca = False
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the coffee roasting dataset with your specific parameters"""
        print("Loading dataset...")
        df = pd.read_csv(filepath)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Extract time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        categorical_feature_cols = ['bean_type', 'target_roast_level']
        for col in categorical_feature_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        df = self._engineer_features(df)
        
        base_feature_cols = [
            'batch_size_kg', 'initial_moisture_pct', 'bean_density_g_cm3', 'bean_size_mm',
            'preheat_temp_c', 'gas_level_pct', 'airflow_rate_pct', 'drum_speed_rpm',
            'roast_duration_min', 'drying_temp_avg_c', 'maillard_temp_avg_c', 
            'development_temp_avg_c', 'final_temp_c', 'first_crack_time_min',
            'second_crack_time_min', 'ambient_temp_c', 'humidity_pct', 
            'heat_rate_c_per_min', 'total_energy_units', 'weight_loss_pct', 
            'final_weight_kg',
            'color_score_agtron', 'aroma_score', 'body_score', 'acidity_score',
            'defects_count',
            'bean_type_encoded', 'target_roast_level_encoded'
        ]
        
        if 'timestamp' in df.columns:
            base_feature_cols.extend(['hour', 'day_of_week', 'month'])
        
        engineered_features = [
            'temp_range', 'energy_efficiency', 'moisture_loss_rate', 
            'crack_timing_ratio', 'roast_intensity', 'environmental_factor', 
            'process_stability', 'weight_change_ratio', 'crack_gap', 
            'quality_composite', 'temp_progression_rate'
        ]
        
        available_features = []
        for col in base_feature_cols + engineered_features:
            if col in df.columns and not df[col].isnull().all():
                available_features.append(col)
        
        self.feature_columns = available_features
        
        X = df[self.feature_columns]
        y_quality = df['overall_quality_score'] if 'overall_quality_score' in df.columns else None
        y_anomaly = df['process_anomaly'] if 'process_anomaly' in df.columns else None
        
        print(f"Features selected: {len(self.feature_columns)}")
        print(f"Final dataset shape: {X.shape}")

        if y_anomaly is not None:
            print(f"\nAnomaly class distribution:")
            print(y_anomaly.value_counts())
            print(f"Anomaly rate: {y_anomaly.mean():.3f}")
        
        if y_quality is not None:
            print(f"\nQuality score statistics:")
            print(f"Mean: {y_quality.mean():.3f}, Std: {y_quality.std():.3f}")
            print(f"Range: {y_quality.min():.1f} - {y_quality.max():.1f}")
        
        return df, X, y_quality, y_anomaly 
    
    def _engineer_features(self, df):
        """Create additional engineered features based on your dataset"""
        
        temp_cols = ['drying_temp_avg_c', 'maillard_temp_avg_c', 'development_temp_avg_c', 'final_temp_c']
        available_temp_cols = [col for col in temp_cols if col in df.columns]
        if len(available_temp_cols) >= 2:
            df['temp_range'] = df[available_temp_cols].max(axis=1) - df[available_temp_cols].min(axis=1)
        else:
            df['temp_range'] = 0
        
        if 'total_energy_units' in df.columns and 'batch_size_kg' in df.columns:
            df['energy_efficiency'] = df['total_energy_units'] / (df['batch_size_kg'] + 1e-6)
        else:
            df['energy_efficiency'] = 0
        
        if 'initial_moisture_pct' in df.columns and 'roast_duration_min' in df.columns:
            estimated_final_moisture = df['initial_moisture_pct'] * (1 - df.get('weight_loss_pct', 15) / 100)
            df['moisture_loss_rate'] = (df['initial_moisture_pct'] - estimated_final_moisture) / (df['roast_duration_min'] + 1e-6)
        else:
            df['moisture_loss_rate'] = 0
        
        if 'first_crack_time_min' in df.columns and 'roast_duration_min' in df.columns:
            df['crack_timing_ratio'] = df['first_crack_time_min'] / (df['roast_duration_min'] + 1e-6)
        else:
            df['crack_timing_ratio'] = 0
        
        if 'heat_rate_c_per_min' in df.columns and 'roast_duration_min' in df.columns:
            df['roast_intensity'] = df['heat_rate_c_per_min'] * df['roast_duration_min']
        else:
            df['roast_intensity'] = 0
        
        if 'ambient_temp_c' in df.columns and 'humidity_pct' in df.columns:
            df['environmental_factor'] = df['ambient_temp_c'] * (1 + df['humidity_pct'] / 100)
        else:
            df['environmental_factor'] = 0
        
        stability_cols = ['gas_level_pct', 'airflow_rate_pct', 'drum_speed_rpm']
        available_stability_cols = [col for col in stability_cols if col in df.columns]
        if available_stability_cols:
            df['process_stability'] = df[available_stability_cols].std(axis=1)
        else:
            df['process_stability'] = 0
        
        if 'batch_size_kg' in df.columns and 'final_weight_kg' in df.columns:
            df['weight_change_ratio'] = df['final_weight_kg'] / (df['batch_size_kg'] + 1e-6)
        else:
            df['weight_change_ratio'] = 1.0
        
        if 'first_crack_time_min' in df.columns and 'second_crack_time_min' in df.columns:
            df['crack_gap'] = df['second_crack_time_min'] - df['first_crack_time_min']
        else:
            df['crack_gap'] = 0
        
        quality_cols = ['aroma_score', 'body_score', 'acidity_score']
        available_quality_cols = [col for col in quality_cols if col in df.columns]
        if available_quality_cols:
            df['quality_composite'] = df[available_quality_cols].mean(axis=1)
        else:
            df['quality_composite'] = 0
        
        if all(col in df.columns for col in ['drying_temp_avg_c', 'maillard_temp_avg_c', 'development_temp_avg_c']):
            temp_diff1 = df['maillard_temp_avg_c'] - df['drying_temp_avg_c']
            temp_diff2 = df['development_temp_avg_c'] - df['maillard_temp_avg_c']
            df['temp_progression_rate'] = (temp_diff1 + temp_diff2) / 2
        else:
            df['temp_progression_rate'] = 0
        
        return df
    
    def train_quality_model(self, X, y):
        if y is None:
            print("No quality labels available, skipping quality model training.")
            return None, None, None
        
        print("Training quality prediction model...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1
        }
        
        self.quality_model = lgb.train(params, train_data, valid_sets=[valid_data],
                                     num_boost_round=1000, callbacks=[lgb.early_stopping(50, verbose=False)])
        
        y_pred = self.quality_model.predict(X_test_scaled, num_iteration=self.quality_model.best_iteration)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Quality Model Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        return X_test_scaled, y_test, y_pred
    
    def train_anomaly_model(self, X, y):
        """Train a single, optimized LightGBM model for anomaly detection."""
        if y is None:
            print("No anomaly labels available, skipping anomaly model training.")
            return None, None, None, None
        
        print("Training optimized anomaly detection model...")
        
        class_counts = y.value_counts()
        print(f"Class distribution: {dict(class_counts)}")
        
        if len(class_counts) < 2 or class_counts.min() < 2:
            print("Not enough samples in minority class for training.")
            return None, None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if not hasattr(self.scaler, 'scale_'):
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Performing feature selection...")
        max_features = min(20, X_train_scaled.shape[1]) # Select up to 20 best features
        self.feature_selector = SelectKBest(score_func=f_classif, k=max_features)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        print("Handling class imbalance with SMOTE...")
        minority_class_count = y_train.value_counts().min()
        k_neighbors = min(5, minority_class_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_train_selected, y_train)
        print(f"After SMOTE - Class distribution: {dict(pd.Series(y_resampled).value_counts())}")
        
        print("Performing hyperparameter tuning for LightGBM...")
        lgbm = lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)

        param_grid = {
            'n_estimators': [100, 150],
            'learning_rate': [0.05],
            'num_leaves': [31],
            'max_depth': [7]
        }
        
        grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, 
                                   cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
        
        grid_search.fit(X_resampled, y_resampled)
        
        print(f"Best parameters found: {grid_search.best_params_}")
        self.anomaly_model = grid_search.best_estimator_
        
        y_pred = self.anomaly_model.predict(X_test_selected)
        y_pred_proba = self.anomaly_model.predict_proba(X_test_selected)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nOptimized Anomaly Detection Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {auc_score:.4f}")
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_test_selected, y_test, y_pred, y_pred_proba
    
    def get_feature_importance(self):
        importance_data = []
        
        if self.quality_model:
            quality_importance = self.quality_model.feature_importance(importance_type='gain')
            for i, importance in enumerate(quality_importance):
                importance_data.append({
                    'feature': self.feature_columns[i],
                    'importance': importance,
                    'model': 'Quality Prediction'
                })
        
        if self.anomaly_model and hasattr(self.anomaly_model, 'feature_importances_'):
            if self.feature_selector is not None and self.feature_selector.get_support().any():
                selected_features = [self.feature_columns[i] for i in self.feature_selector.get_support(indices=True)]
            else:
                selected_features = self.feature_columns
                
            anomaly_importance = self.anomaly_model.feature_importances_
            for i, importance in enumerate(anomaly_importance):
                if i < len(selected_features):  # Safety check
                    importance_data.append({
                        'feature': selected_features[i],
                        'importance': importance,
                        'model': 'Anomaly Detection'
                    })
        
        return pd.DataFrame(importance_data)
    
    def predict(self, X):
        if not self.anomaly_model:
            raise ValueError("Anomaly model not trained yet!")
        
        X_features = X[self.feature_columns]
        X_scaled = self.scaler.transform(X_features)
        
        quality_pred = None
        if self.quality_model:
            quality_pred = self.quality_model.predict(X_scaled, num_iteration=self.quality_model.best_iteration)
        
        if self.feature_selector is not None and self.feature_selector.get_support().any():
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
            
        anomaly_pred = self.anomaly_model.predict_proba(X_selected)[:, 1]
        
        return quality_pred, anomaly_pred
    
    def save_models(self, quality_path='enhanced_coffee_quality_model.pkl', 
                   anomaly_path='enhanced_coffee_anomaly_model.pkl', 
                   scaler_path='enhanced_coffee_scaler.pkl'):
        """Save trained models and preprocessors"""
        if self.quality_model:
            joblib.dump(self.quality_model, quality_path)
        if self.anomaly_model:
            joblib.dump(self.anomaly_model, anomaly_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump({
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'use_pca': self.use_pca
        }, 'enhanced_coffee_preprocessors.pkl')
        print(f"Enhanced models saved successfully!")
    
    def load_models(self, quality_path='enhanced_coffee_quality_model.pkl',
                   anomaly_path='enhanced_coffee_anomaly_model.pkl',
                   scaler_path='enhanced_coffee_scaler.pkl'):

        try:
            self.quality_model = joblib.load(quality_path)
        except FileNotFoundError:
            print("Quality model not found")
        
        try:
            self.anomaly_model = joblib.load(anomaly_path)
        except FileNotFoundError:
            print("Anomaly model not found")
        
        self.scaler = joblib.load(scaler_path)
        preprocessors = joblib.load('enhanced_coffee_preprocessors.pkl')
        self.label_encoders = preprocessors['label_encoders']
        self.feature_columns = preprocessors['feature_columns']
        self.feature_selector = preprocessors['feature_selector']
        self.pca = preprocessors.get('pca')
        self.use_pca = preprocessors.get('use_pca', False)
        print("Enhanced models loaded successfully!")

def main():
    predictor = EnhancedCoffeeRoastPredictor()
    
    try:
        df, X, y_quality, y_anomaly = predictor.load_and_preprocess_data('FNB_Coffee_Roast_Dataset.csv')
    except FileNotFoundError:
        print("Dataset file 'FNB_Coffee_Roast_Dataset.csv' not found!")
        print("Please ensure the file is in the same directory as this script.")
        return
    
    if y_quality is not None:
        X_test_q, y_test_q, y_pred_q = predictor.train_quality_model(X, y_quality)
    else:
        X_test_q, y_test_q, y_pred_q = None, None, None
        print("No 'overall_quality_score' column found for quality prediction.")
    
    if y_anomaly is not None:
        result = predictor.train_anomaly_model(X, y_anomaly)
        if result and result[0] is not None:
            X_test_a, y_test_a, y_pred_a, y_pred_proba_a = result
        else:
            print("Anomaly model training failed or was skipped.")
            X_test_a, y_test_a, y_pred_a, y_pred_proba_a = (None, None, None, None)
    else:
        print("No 'process_anomaly' column found for anomaly detection.")
        return
    
    importance_df = predictor.get_feature_importance()
    anomaly_importance = pd.DataFrame()
    quality_importance = pd.DataFrame()
    if not importance_df.empty:
        print("\nTop 10 Most Important Features for Anomaly Detection:")
        anomaly_importance = importance_df[importance_df['model'] == 'Anomaly Detection'].sort_values('importance', ascending=False)
        quality_importance = importance_df[importance_df['model'] == 'Quality Prediction'].sort_values('importance', ascending=False)
        if not anomaly_importance.empty:
            print(anomaly_importance.head(10))

    predictor.save_models()
    
    plt.figure(figsize=(20, 15))
    
    plt.subplot(3, 4, 1)
    if y_test_a is not None and y_pred_a is not None:
        cm = confusion_matrix(y_test_a, y_pred_a)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Enhanced Anomaly Detection\nConfusion Matrix')
    else:
        plt.text(0.5, 0.5, 'Not Available', ha='center')
        plt.title('Confusion Matrix')

    plt.subplot(3, 4, 2)
    if y_test_a is not None and y_pred_proba_a is not None:
        try:
            from sklearn.metrics import roc_curve 
            fpr, tpr, _ = roc_curve(y_test_a, y_pred_proba_a)
            auc_score = roc_auc_score(y_test_a, y_pred_proba_a)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
        except ValueError:
            plt.text(0.5, 0.5, 'ROC Curve\nnot available', ha='center', va='center')
            plt.title('ROC Curve')
    else:
        plt.text(0.5, 0.5, 'Not Available', ha='center')
        plt.title('ROC Curve')

    plt.subplot(3, 4, 3)
    if y_test_a is not None and y_pred_proba_a is not None:
        try:
            precision, recall, _ = precision_recall_curve(y_test_a, y_pred_proba_a)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
        except ValueError:
            plt.text(0.5, 0.5, 'PR Curve\nnot available', ha='center', va='center')
            plt.title('Precision-Recall Curve')
    else:
        plt.text(0.5, 0.5, 'Not Available', ha='center')
        plt.title('Precision-Recall Curve')

    plt.subplot(3, 4, 4)
    if y_test_a is not None and y_pred_proba_a is not None:
        try:
            sns.histplot(y_pred_proba_a[y_test_a == 0], bins=20, alpha=0.7, label='Normal', color='blue', stat='density', kde=True)
            sns.histplot(y_pred_proba_a[y_test_a == 1], bins=20, alpha=0.7, label='Anomaly', color='red', stat='density', kde=True)
            plt.xlabel('Anomaly Probability')
            plt.ylabel('Density')
            plt.title('Anomaly Score Distribution')
            plt.legend()
        except Exception:
            plt.text(0.5, 0.5, 'Score Distribution\nnot available', ha='center', va='center')
            plt.title('Anomaly Score Distribution')
    else:
        plt.text(0.5, 0.5, 'Not Available', ha='center')
        plt.title('Anomaly Score Distribution')

    plt.subplot(3, 4, 5)
    if not anomaly_importance.empty:
        top_anomaly_features = anomaly_importance.head(10)
        sns.barplot(x='importance', y='feature', data=top_anomaly_features, palette='viridis')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.title('Top 10 Features for Anomaly Detection')
    else:
        plt.text(0.5, 0.5, 'Not Available', ha='center')
        plt.title('Feature Importance (Anomaly)')

    plt.subplot(3, 4, 6)
    if y_test_q is not None and y_pred_q is not None:
        plt.scatter(y_test_q, y_pred_q, alpha=0.6, edgecolor='k')
        plt.plot([y_test_q.min(), y_test_q.max()], [y_test_q.min(), y_test_q.max()], 'r--', lw=2)
        plt.xlabel('Actual Quality Score')
        plt.ylabel('Predicted Quality Score')
        plt.title('Quality Prediction: Actual vs. Predicted')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Not Available', ha='center')
        plt.title('Quality Prediction')

    plt.subplot(3, 4, 7)
    if y_test_q is not None and y_pred_q is not None:
        residuals = y_test_q - y_pred_q
        sns.residplot(x=y_pred_q, y=residuals, lowess=False, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 2})
        plt.xlabel('Predicted Quality Score')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.title('Quality Model Residual Plot')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Not Available', ha='center')
        plt.title('Residual Plot')

    plt.subplot(3, 4, 8)
    if not quality_importance.empty:
        top_quality_features = quality_importance.head(10)
        sns.barplot(x='importance', y='feature', data=top_quality_features, palette='plasma')
        plt.xlabel('Importance Score (Gain)')
        plt.ylabel('Feature')
        plt.title('Top 10 Features for Quality Prediction')
    else:
        plt.text(0.5, 0.5, 'Not Available', ha='center')
        plt.title('Feature Importance (Quality)')

    plt.subplot(3, 4, 9)
    numeric_X = X.select_dtypes(include=np.number)
    corr = numeric_X.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Feature Correlation Matrix')
    
    plt.subplot(3, 4, 10)
    if y_quality is not None and 'bean_type' in df.columns:
        temp_df = pd.DataFrame({'quality': y_quality, 'bean_type': df['bean_type']})
        sns.boxplot(x='bean_type', y='quality', data=temp_df)
        plt.title('Quality Score by Bean Type')
    else:
        plt.text(0.5, 0.5, 'Not Available', ha='center')
        plt.title('Quality Score by Bean Type')

    plt.subplot(3, 4, 11)
    if y_anomaly is not None and 'target_roast_level' in df.columns:
        temp_df = pd.DataFrame({'anomaly': y_anomaly, 'roast_level': df['target_roast_level']})
        sns.barplot(x='roast_level', y='anomaly', data=temp_df)
        plt.ylabel('Anomaly Rate')
        plt.title('Anomaly Rate by Roast Level')
    else:
        plt.text(0.5, 0.5, 'Not Available', ha='center')
        plt.title('Anomaly Rate by Roast Level')

    plt.subplot(3, 4, 12)
    plt.axis('off') 

    plt.tight_layout()
    plt.savefig('enhanced_model_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nEnhanced model training completed successfully!")
    print("Files generated:")
    print("- enhanced_coffee_quality_model.pkl")
    print("- enhanced_coffee_anomaly_model.pkl") 
    print("- enhanced_coffee_scaler.pkl")
    print("- enhanced_coffee_preprocessors.pkl")
    print("- enhanced_model_evaluation_results.png")

if __name__ == "__main__":
    main()
