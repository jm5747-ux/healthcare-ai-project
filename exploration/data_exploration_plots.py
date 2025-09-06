import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


def plot_exploration_graphs(data):
    """
    Plot the data exploration plots
    """

    # Create a comprehensive data exploration
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hospital Readmission Data Analysis', fontsize=16, fontweight='bold')

    # 1. Age distribution by readmission status
    axes[0, 0].hist([data[data['readmitted_30_days']==0]['age'], 
                    data[data['readmitted_30_days']==1]['age']], 
                    bins=20, alpha=0.7, label=['No Readmission', 'Readmission'])
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Age Distribution by Readmission Status')
    axes[0, 0].legend()

    # 2. Readmission rate by diagnosis
    diagnosis_readmission = data.groupby('primary_diagnosis')['readmitted_30_days'].mean().sort_values(ascending=False)
    diagnosis_readmission.plot(kind='bar', ax=axes[0, 1], color='skyblue')
    axes[0, 1].set_xlabel('Primary Diagnosis')
    axes[0, 1].set_ylabel('Readmission Rate')
    axes[0, 1].set_title('Readmission Rate by Diagnosis')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Length of stay vs readmission
    axes[0, 2].boxplot([data[data['readmitted_30_days']==0]['length_of_stay'], 
                        data[data['readmitted_30_days']==1]['length_of_stay']], 
                    labels=['No Readmission', 'Readmission'])
    axes[0, 2].set_ylabel('Length of Stay (days)')
    axes[0, 2].set_title('Length of Stay by Readmission Status')

    # 4. Number of medications
    axes[1, 0].hist([data[data['readmitted_30_days']==0]['num_medications'], 
                    data[data['readmitted_30_days']==1]['num_medications']], 
                    bins=15, alpha=0.7, label=['No Readmission', 'Readmission'])
    axes[1, 0].set_xlabel('Number of Medications')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Medication Count by Readmission Status')
    axes[1, 0].legend()

    # 5. Number of procedures
    axes[1, 1].hist([data[data['readmitted_30_days']==0]['num_procedures'], 
                    data[data['readmitted_30_days']==1]['num_procedures']], 
                    bins=10, alpha=0.7, label=['No Readmission', 'Readmission'])
    axes[1, 1].set_xlabel('Number of Procedures')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Procedure Count by Readmission Status')
    axes[1, 1].legend()

    # 6. Emergency admission vs readmission
    axes[1, 2].boxplot([data[data['readmitted_30_days']==0]['emergency_admission'], 
                    data[data['readmitted_30_days']==1]['emergency_admission']], 
                    labels=['No Readmission', 'Readmission'])
    axes[1, 2].set_ylabel('Emergency Admission')
    axes[1, 2].set_title('Emergency Admission by Readmission Status')

    # Show the plots
    plt.tight_layout()
    plt.show()          