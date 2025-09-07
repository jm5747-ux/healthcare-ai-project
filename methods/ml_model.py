import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

class HospitalReadmissionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def preprocess_data(self, data):
        """Preprocess the data for ML model"""
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Encode categorical variables
        categorical_columns = ['gender', 'primary_diagnosis', 'insurance_type', 'discharge_destination']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Select features for the model (including new clinical features)
        feature_columns = [
            'age', 'gender', 'primary_diagnosis', 'length_of_stay', 
            'num_medications', 'num_procedures', 'emergency_admission',
            'insurance_type', 'prev_hospitalizations', 'chronic_conditions',
            'discharge_destination', 'hemoglobin', 'creatinine', 'glucose',
            'comorbidity_score', 'recent_readmission', 'albumin', 'wbc',
            'follow_up', 'discharge_instructions'
        ]
        
        self.feature_columns = feature_columns
        
        # Prepare features and target
        X = df[feature_columns]
        y = df['readmitted_30_days']
        
        return X, y
    
    def train_model(self, X, y, model_type='random_forest'):
        """Train the ML model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose and train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        return X_test_scaled, y_test, y_pred, y_pred_proba
    
    def evaluate_model(self, X_test, y_test, y_pred, y_pred_proba):
        """Evaluate the model performance"""
        print("=== Model Evaluation ===")
        print(f"Accuracy: {(y_pred == y_test).mean():.3f}")
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=['No Readmission', 'Readmission']))
        
        print("\n=== Confusion Matrix ===")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Readmission', 'Readmission'],
                   yticklabels=['No Readmission', 'Readmission'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('roc_curve.png')
        plt.show()
    
    def feature_importance(self):
        """Display feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("\n=== Feature Importance ===")
            print(feature_importance_df)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance_df, x='importance', y='feature')
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.show()
    
    def predict_new_patient(self, patient_data):
        """Predict readmission risk for a new patient"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess the new patient data
        patient_df = pd.DataFrame([patient_data])
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in patient_df.columns:
                patient_df[col] = le.transform(patient_df[col])
        
        # Select features
        X_new = patient_df[self.feature_columns]
        
        # Scale features
        X_new_scaled = self.scaler.transform(X_new)
        
        # Make prediction
        prediction = self.model.predict(X_new_scaled)[0]
        probability = self.model.predict_proba(X_new_scaled)[0][1]
        
        return prediction, probability
