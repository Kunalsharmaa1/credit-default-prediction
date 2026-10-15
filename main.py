"""
================================================================================
CREDIT DEFAULT PREDICTION USING LOGISTIC REGRESSION AND LINEAR DISCRIMINANT ANALYSIS
================================================================================
Academic Project: Predictive Analytics
Dataset: UCI Default of Credit Card Clients (Dataset ID: 350)
Methodology: Logistic Regression & Linear Discriminant Analysis (LDA)
Scope: Strict syllabus compliance (No tree ensembles, neural nets, or out-of-scope models)
Author: Student Predictive Analytics Team
================================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Suppress minor warnings for clean presentation
warnings.filterwarnings('ignore')

# Set aesthetic style for all plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Fixed random state for strict reproducibility across all operations
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Directory configurations
DATA_DIR = 'data'
RESULTS_DIR = 'results'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
# 1. DATA INGESTION & LOCAL CACHING
# ==============================================================================
def load_dataset():
    """
    Reliably loads the UCI Default of Credit Card Clients dataset (ID 350).
    Checks local cache first. If not found, fetches from ucimlrepo and caches locally.
    """
    cache_path = os.path.join(DATA_DIR, 'default_of_credit_card_clients.csv')
    
    print("=" * 80)
    print("[STEP 1/7] INGESTING DATASET (UCI ID: 350)")
    print("=" * 80)
    
    if os.path.exists(cache_path):
        print(f"-> Found local cached dataset at '{cache_path}'. Loading from disk...")
        df_raw = pd.read_csv(cache_path)
    else:
        print("-> Fetching dataset ID 350 from UCI Machine Learning Repository...")
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=350)
            X = dataset.data.features
            y = dataset.data.targets
            df_raw = pd.concat([X, y], axis=1)
            df_raw.to_csv(cache_path, index=False)
            print(f"-> Successfully downloaded and cached {len(df_raw)} records to '{cache_path}'.")
        except Exception as e:
            print(f"[ERROR] Failed to fetch dataset: {e}")
            sys.exit(1)
            
    print(f"-> Raw Dataset Dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
    return df_raw


# ==============================================================================
# 2. DATA CLEANING, SEMANTIC RENAMING & PREPROCESSING
# ==============================================================================
def clean_and_harmonize_data(df_raw):
    """
    Renames abstract column names (X1..X23, Y) to domain-relevant financial names,
    inspects missing values, removes duplicate records, and harmonizes undocumented categories.
    """
    print("\n" + "=" * 80)
    print("[STEP 2/7] DATA INSPECTION, CLEANING & SCHEMA HARMONIZATION")
    print("=" * 80)
    
    df = df_raw.copy()
    
    # Semantic mapping from UCI codes to financial feature names
    column_mapping = {
        'X1': 'LIMIT_BAL',
        'X2': 'SEX',
        'X3': 'EDUCATION',
        'X4': 'MARRIAGE',
        'X5': 'AGE',
        'X6': 'PAY_0',
        'X7': 'PAY_2',
        'X8': 'PAY_3',
        'X9': 'PAY_4',
        'X10': 'PAY_5',
        'X11': 'PAY_6',
        'X12': 'BILL_AMT1',
        'X13': 'BILL_AMT2',
        'X14': 'BILL_AMT3',
        'X15': 'BILL_AMT4',
        'X16': 'BILL_AMT5',
        'X17': 'BILL_AMT6',
        'X18': 'PAY_AMT1',
        'X19': 'PAY_AMT2',
        'X20': 'PAY_AMT3',
        'X21': 'PAY_AMT4',
        'X22': 'PAY_AMT5',
        'X23': 'PAY_AMT6',
        'Y': 'DEFAULT'
    }
    
    # Rename if UCI naming convention is present
    df.rename(columns=column_mapping, inplace=True)
    if 'default payment next month' in df.columns:
        df.rename(columns={'default payment next month': 'DEFAULT'}, inplace=True)
        
    print("-> Features successfully mapped to human-readable domain names:")
    print("   LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0..6, BILL_AMT1..6, PAY_AMT1..6, DEFAULT")
    
    # Missing Value Inspection
    missing_count = df.isnull().sum().sum()
    print(f"-> Total Missing Values Detected: {missing_count}")
    
    # Duplicate Record Inspection & De-duplication
    duplicates = df.duplicated().sum()
    print(f"-> Exact Duplicate Rows Identified: {duplicates}")
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"-> Removed {duplicates} duplicate entries. Preserved {len(df)} distinct customer profiles.")
        
    # Categorical Harmonization:
    # EDUCATION: 1 = Graduate School, 2 = University, 3 = High School, 4 = Others
    # Levels 0, 5, 6 are undocumented/unknown categories. Merge them into 4 ('Others').
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x in [0, 4, 5, 6] else x)
    
    # MARRIAGE: 1 = Married, 2 = Single, 3 = Others
    # Level 0 is undocumented. Merge into 3 ('Others').
    df['MARRIAGE'] = df['MARRIAGE'].apply(lambda x: 3 if x in [0, 3] else x)
    
    print("-> Cleaned undocumented categorical levels in EDUCATION (0, 5, 6 -> 4) and MARRIAGE (0 -> 3).")
    
    # Target Class Distribution Inspection
    target_counts = df['DEFAULT'].value_counts()
    target_pcts = df['DEFAULT'].value_counts(normalize=True) * 100
    print(f"-> Target Class Distribution: Non-Default (0): {target_counts[0]} ({target_pcts[0]:.2f}%), "
          f"Default (1): {target_counts[1]} ({target_pcts[1]:.2f}%)")
    print("   [KEY OBSERVATION] The dataset exhibits moderate class imbalance (~78% vs ~22%).")
    print("   A naive model predicting zero default would attain ~78% accuracy while failing 100% of risk detection!")
    
    return df


# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================
def engineer_features(df):
    """
    Constructs domain-specific financial credit risk metrics:
    1. PAY_MAX: Maximum repayment delay across all 6 observation months.
    2. BILL_AMT_AVG: Mean monthly statement balance.
    3. PAY_AMT_AVG: Mean monthly payment made.
    4. UTILIZATION_1: Credit utilization ratio in month 1 (BILL_AMT1 / LIMIT_BAL).
    5. PAY_TO_BILL_RATIO_1: Payment compliance ratio in month 1 (PAY_AMT1 / (BILL_AMT2 + 1)).
    """
    print("\n" + "=" * 80)
    print("[STEP 3/7] DOMAIN FEATURE ENGINEERING")
    print("=" * 80)
    
    df_feat = df.copy()
    
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # 1. Maximum delinquency level across the 6-month observation window
    df_feat['PAY_MAX'] = df_feat[pay_cols].max(axis=1)
    
    # 2. Average bill balance over 6 months
    df_feat['BILL_AMT_AVG'] = df_feat[bill_cols].mean(axis=1)
    
    # 3. Average payment amount over 6 months
    df_feat['PAY_AMT_AVG'] = df_feat[pay_amt_cols].mean(axis=1)
    
    # 4. Credit utilization ratio for the most recent statement month (September 2005)
    df_feat['UTILIZATION_1'] = df_feat['BILL_AMT1'] / (df_feat['LIMIT_BAL'] + 1.0)
    df_feat['UTILIZATION_1'] = df_feat['UTILIZATION_1'].clip(lower=-1.0, upper=5.0)
    
    # 5. Payment compliance ratio (Payment in Sept relative to August statement balance)
    df_feat['PAY_TO_BILL_RATIO_1'] = df_feat['PAY_AMT1'] / (df_feat['BILL_AMT2'].abs() + 1.0)
    df_feat['PAY_TO_BILL_RATIO_1'] = df_feat['PAY_TO_BILL_RATIO_1'].clip(lower=0.0, upper=10.0)
    
    print("-> Engineered 5 domain features:")
    print("   1. PAY_MAX (Max delinquency delay across 6 months)")
    print("   2. BILL_AMT_AVG (6-month average bill statement amount)")
    print("   3. PAY_AMT_AVG (6-month average payment amount)")
    print("   4. UTILIZATION_1 (Recent credit limit utilization ratio)")
    print("   5. PAY_TO_BILL_RATIO_1 (Recent payment compliance ratio)")
    print(f"-> Total features after engineering: {df_feat.shape[1] - 1} predictive variables.")
    
    return df_feat


# ==============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
# ==============================================================================
def perform_eda(df):
    """
    Generates publication-quality EDA figures saved directly to results/ directory.
    """
    print("\n" + "=" * 80)
    print("[STEP 4/7] GENERATING EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS")
    print("=" * 80)
    
    # Plot 1: Target Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    counts = df['DEFAULT'].value_counts()
    pcts = df['DEFAULT'].value_counts(normalize=True) * 100
    bars = ax.bar(['Non-Default (0)', 'Default (1)'], counts, color=['#2b5c8f', '#d95f02'], width=0.5, edgecolor='black', alpha=0.85)
    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 400,
                f'{height:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_title('Target Class Distribution (Default vs Non-Default)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Clients', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p1 = os.path.join(RESULTS_DIR, '01_target_distribution.png')
    plt.savefig(p1)
    plt.close()
    print(f"-> Saved: {p1}")

    # Plot 2: Default Rate by Demographic Segments
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    
    sex_map = {1: 'Male', 2: 'Female'}
    sex_default = df.groupby('SEX')['DEFAULT'].mean() * 100
    axes[0, 0].bar([sex_map[k] for k in sex_default.index], sex_default.values, color=['#4575b4', '#d73027'], edgecolor='black', alpha=0.85)
    axes[0, 0].set_title('Default Rate by Gender', fontweight='bold', fontsize=11)
    axes[0, 0].set_ylabel('Default Rate (%)')
    for i, v in enumerate(sex_default.values):
        axes[0, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    axes[0, 0].set_ylim(0, max(sex_default.values) * 1.25)
    
    edu_map = {1: 'Grad School', 2: 'University', 3: 'High School', 4: 'Others'}
    edu_default = df.groupby('EDUCATION')['DEFAULT'].mean() * 100
    axes[0, 1].bar([edu_map[k] for k in edu_default.index], edu_default.values, color='#313695', edgecolor='black', alpha=0.85)
    axes[0, 1].set_title('Default Rate by Education Level', fontweight='bold', fontsize=11)
    axes[0, 1].set_ylabel('Default Rate (%)')
    for i, v in enumerate(edu_default.values):
        axes[0, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    axes[0, 1].set_ylim(0, max(edu_default.values) * 1.25)
    
    marr_map = {1: 'Married', 2: 'Single', 3: 'Others'}
    marr_default = df.groupby('MARRIAGE')['DEFAULT'].mean() * 100
    axes[1, 0].bar([marr_map[k] for k in marr_default.index], marr_default.values, color='#74add1', edgecolor='black', alpha=0.85)
    axes[1, 0].set_title('Default Rate by Marital Status', fontweight='bold', fontsize=11)
    axes[1, 0].set_ylabel('Default Rate (%)')
    for i, v in enumerate(marr_default.values):
        axes[1, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    axes[1, 0].set_ylim(0, max(marr_default.values) * 1.25)
    
    df['AGE_BIN'] = pd.cut(df['AGE'], bins=[20, 30, 40, 50, 60, 80], labels=['20-29', '30-39', '40-49', '50-59', '60+'])
    age_default = df.groupby('AGE_BIN', observed=False)['DEFAULT'].mean() * 100
    axes[1, 1].bar(age_default.index.astype(str), age_default.values, color='#fdae61', edgecolor='black', alpha=0.85)
    axes[1, 1].set_title('Default Rate by Age Bracket', fontweight='bold', fontsize=11)
    axes[1, 1].set_ylabel('Default Rate (%)')
    for i, v in enumerate(age_default.values):
        axes[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    axes[1, 1].set_ylim(0, max(age_default.values) * 1.25)
    df.drop(columns=['AGE_BIN'], inplace=True)
    
    plt.suptitle('Default Rate Across Demographic Segments', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    p2 = os.path.join(RESULTS_DIR, '02_demographic_default_rates.png')
    plt.savefig(p2)
    plt.close()
    print(f"-> Saved: {p2}")

    # Plot 3: Repayment Status (PAY_0) Impact on Default
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    pay0_default = df.groupby('PAY_0')['DEFAULT'].agg(['mean', 'count'])
    pay0_default['mean_pct'] = pay0_default['mean'] * 100
    
    bars = ax.bar([str(x) for x in pay0_default.index], pay0_default['mean_pct'], color='#e41a1c', edgecolor='black', alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1.2, f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('Default Rate by Recent Repayment Status (PAY_0: Sept 2005)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('PAY_0 Status Code (-2: No consumption, -1: Paid duly, 0: Revolving credit, 1..8: Months Delayed)', fontsize=10)
    ax.set_ylabel('Observed Default Rate (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p3 = os.path.join(RESULTS_DIR, '03_repayment_status_impact.png')
    plt.savefig(p3)
    plt.close()
    print(f"-> Saved: {p3}")

    # Plot 4: Credit Limit Distribution by Default Status
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.boxplot(x='DEFAULT', y='LIMIT_BAL', data=df, palette=['#2b5c8f', '#d95f02'], ax=ax, width=0.4, showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    ax.set_xticklabels(['Non-Default (0)', 'Default (1)'], fontsize=11)
    ax.set_title('Credit Limit Balance (LIMIT_BAL in NT$) by Default Status', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Credit Limit (NT$)', fontsize=11)
    ax.set_xlabel('Default Payment Status', fontsize=11)
    plt.tight_layout()
    p4 = os.path.join(RESULTS_DIR, '04_credit_limit_distribution.png')
    plt.savefig(p4)
    plt.close()
    print(f"-> Saved: {p4}")

    # Plot 5: Correlation Heatmap of Key Features
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    selected_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_MAX', 'BILL_AMT1', 'BILL_AMT_AVG',
                     'PAY_AMT1', 'PAY_AMT_AVG', 'UTILIZATION_1', 'PAY_TO_BILL_RATIO_1', 'DEFAULT']
    corr_matrix = df[selected_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.4, vmax=0.8,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix of Key Credit & Delinquency Features', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    p5 = os.path.join(RESULTS_DIR, '05_correlation_heatmap.png')
    plt.savefig(p5)
    plt.close()
    print(f"-> Saved: {p5}")


# ==============================================================================
# 5. STRATIFIED TRAIN/TEST SPLIT & PIPELINE PREPROCESSING
# ==============================================================================
def prepare_data_and_pipeline(df):
    """
    Performs a leak-free 80/20 stratified split.
    Constructs a ColumnTransformer to apply StandardScaler to numerical features
    and OneHotEncoder to categorical features. Preprocessor is fitted strictly on train data.
    """
    print("\n" + "=" * 80)
    print("[STEP 5/7] STRATIFIED DATA SPLIT & PREPROCESSING PIPELINE SETUP")
    print("=" * 80)
    
    X = df.drop(columns=['DEFAULT'])
    y = df['DEFAULT']
    
    # Stratified Train/Test Split (80% Train, 20% Untouched Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"-> Training Split: {X_train.shape[0]} samples ({y_train.mean()*100:.2f}% default rate)")
    print(f"-> Testing Split:  {X_test.shape[0]} samples ({y_test.mean()*100:.2f}% default rate)")
    print("-> Stratification confirmed: Class proportions strictly preserved across splits.")
    
    # Feature grouping
    categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    numerical_cols = [c for c in X.columns if c not in categorical_cols]
    
    print(f"-> Categorical Features ({len(categorical_cols)}): {categorical_cols}")
    print(f"-> Numerical Features ({len(numerical_cols)}): {len(numerical_cols)} features (scaled via StandardScaler)")
    
    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    return X_train, X_test, y_train, y_test, preprocessor


# ==============================================================================
# 6. MODEL TRAINING & 5-FOLD CROSS-VALIDATION HYPERPARAMETER TUNING
# ==============================================================================
def train_and_tune_models(X_train, y_train, preprocessor):
    """
    Trains and tunes Logistic Regression and Linear Discriminant Analysis (LDA)
    using Stratified 5-Fold Cross Validation.
    """
    print("\n" + "=" * 80)
    print("[STEP 6/7] MODEL TRAINING & 5-FOLD CV HYPERPARAMETER TUNING")
    print("=" * 80)
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Model 1: Logistic Regression
    print("-> Tuning Logistic Regression via 5-Fold Stratified Cross-Validation (Metric: ROC-AUC)...")
    pipe_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])
    
    param_grid_lr = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__solver': ['lbfgs'],
        'classifier__class_weight': [None, 'balanced']
    }
    
    grid_lr = GridSearchCV(
        estimator=pipe_lr,
        param_grid=param_grid_lr,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        return_train_score=True
    )
    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_
    
    print(f"   Best Logistic Regression Parameters: {grid_lr.best_params_}")
    print(f"   Best 5-Fold CV ROC-AUC: {grid_lr.best_score_:.4f}")
    
    # Model 2: Linear Discriminant Analysis (LDA)
    print("\n-> Tuning Linear Discriminant Analysis (LDA) via 5-Fold Stratified Cross-Validation (Metric: ROC-AUC)...")
    pipe_lda = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LinearDiscriminantAnalysis())
    ])
    
    param_grid_lda = [
        {'classifier__solver': ['svd'], 'classifier__shrinkage': [None]},
        {'classifier__solver': ['lsqr'], 'classifier__shrinkage': [None, 'auto']}
    ]
    
    grid_lda = GridSearchCV(
        estimator=pipe_lda,
        param_grid=param_grid_lda,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        return_train_score=True
    )
    grid_lda.fit(X_train, y_train)
    best_lda = grid_lda.best_estimator_
    
    print(f"   Best LDA Parameters: {grid_lda.best_params_}")
    print(f"   Best 5-Fold CV ROC-AUC: {grid_lda.best_score_:.4f}")
    
    return {
        'Logistic Regression': (best_lr, grid_lr),
        'Linear Discriminant Analysis': (best_lda, grid_lda)
    }


# ==============================================================================
# 7. EVALUATION, THRESHOLD TUNING, OVERFITTING ANALYSIS & REPORTING
# ==============================================================================
def evaluate_and_compare(models_dict, X_train, y_train, X_test, y_test):
    """
    Evaluates models on untouched test split, analyzes overfitting/underfitting,
    tunes decision thresholds, and generates summary tables and comparison figures.
    """
    print("\n" + "=" * 80)
    print("[STEP 7/7] MODEL EVALUATION, COMPARISON & RISK ANALYSIS")
    print("=" * 80)
    
    results = []
    overfitting_data = []
    eval_predictions = {}
    
    for name, (model, grid) in models_dict.items():
        # Train & CV Scores for Overfitting Analysis
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        train_acc = accuracy_score(y_train, train_preds)
        train_auc = roc_auc_score(y_train, train_probs)
        cv_auc = grid.best_score_
        
        # Test Scores
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, test_preds)
        prec = precision_score(y_test, test_preds, zero_division=0)
        rec = recall_score(y_test, test_preds, zero_division=0)
        f1 = f1_score(y_test, test_preds, zero_division=0)
        auc = roc_auc_score(y_test, test_probs)
        
        cm = confusion_matrix(y_test, test_preds)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        
        eval_predictions[name] = {
            'y_pred': test_preds,
            'y_prob': test_probs,
            'cm': cm
        }
        
        results.append({
            'Model': name,
            'Best Hyperparameters': str(grid.best_params_),
            '5-Fold CV ROC-AUC': round(cv_auc, 4),
            'Test Accuracy': round(acc, 4),
            'Test Precision': round(prec, 4),
            'Test Recall': round(rec, 4),
            'Test Specificity': round(specificity, 4),
            'Test F1-Score': round(f1, 4),
            'Test ROC-AUC': round(auc, 4),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        })
        
        overfitting_data.append({
            'Model': name,
            'Train Accuracy': round(train_acc, 4),
            'Test Accuracy': round(acc, 4),
            'Train ROC-AUC': round(train_auc, 4),
            'CV ROC-AUC (5-Fold)': round(cv_auc, 4),
            'Test ROC-AUC': round(auc, 4),
            'AUC Generalization Gap (Train - Test)': round(train_auc - auc, 4)
        })
        
        print(f"\n--- Detailed Classification Report: {name} (Threshold = 0.5) ---")
        print(classification_report(y_test, test_preds, target_names=['Non-Default (0)', 'Default (1)'], digits=4))
        print(f"Confusion Matrix:\nTN: {tn:,} | FP: {fp:,}\nFN: {fn:,} | TP: {tp:,}")
        
    df_results = pd.DataFrame(results)
    df_overfitting = pd.DataFrame(overfitting_data)
    
    # Save results to CSV & Markdown
    csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    md_path = os.path.join(RESULTS_DIR, 'model_comparison.md')
    df_results.to_csv(csv_path, index=False)
    
    # Write Markdown summary table with robust fallback
    def df_to_md(df_in):
        try:
            return df_in.to_markdown(index=False)
        except Exception:
            headers = "| " + " | ".join(df_in.columns) + " |"
            sep = "| " + " | ".join(["---"] * len(df_in.columns)) + " |"
            rows = ["| " + " | ".join(str(val) for val in row) + " |" for row in df_in.values]
            return "\n".join([headers, sep] + rows)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Predictive Analytics Model Comparison Table\n\n")
        f.write(df_to_md(df_results))
        f.write("\n\n## Overfitting and Underfitting Analysis\n\n")
        f.write(df_to_md(df_overfitting))
        
    print(f"\n-> Saved evaluation tables to '{csv_path}' and '{md_path}'.")
    
    # Plot 6: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for idx, (name, data) in enumerate(eval_predictions.items()):
        cm = data['cm']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        labels = np.array([
            [f"{cm[0,0]:,}\n({cm_norm[0,0]*100:.1f}%)\nTN", f"{cm[0,1]:,}\n({cm_norm[0,1]*100:.1f}%)\nFP"],
            [f"{cm[1,0]:,}\n({cm_norm[1,0]*100:.1f}%)\nFN", f"{cm[1,1]:,}\n({cm_norm[1,1]*100:.1f}%)\nTP"]
        ])
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Pred: No Default', 'Pred: Default'],
                    yticklabels=['Actual: No Default', 'Actual: Default'], annot_kws={"fontsize": 11})
        axes[idx].set_title(f"{name}\nConfusion Matrix (t = 0.5)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    p6 = os.path.join(RESULTS_DIR, '06_confusion_matrices.png')
    plt.savefig(p6)
    plt.close()
    print(f"-> Saved: {p6}")

    # Plot 7: ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    colors = ['#2b5c8f', '#d95f02']
    for (name, data), color in zip(eval_predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
        auc_val = roc_auc_score(y_test, data['y_prob'])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.5000)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    p7 = os.path.join(RESULTS_DIR, '07_roc_curves.png')
    plt.savefig(p7)
    plt.close()
    print(f"-> Saved: {p7}")

    # Plot 8: Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    for (name, data), color in zip(eval_predictions.items(), colors):
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, data['y_prob'])
        ax.plot(rec_curve, prec_curve, label=f"{name}", color=color, linewidth=2)
    baseline = y_test.mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline Default Rate ({baseline:.3f})')
    ax.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Recall (Detection Rate of Defaulters)', fontsize=11)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=11)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    plt.tight_layout()
    p8 = os.path.join(RESULTS_DIR, '08_precision_recall_curves.png')
    plt.savefig(p8)
    plt.close()
    print(f"-> Saved: {p8}")

    # Plot 9: Threshold Tuning Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    thresholds = np.linspace(0.1, 0.9, 81)
    threshold_results = {}
    
    for idx, (name, data) in enumerate(eval_predictions.items()):
        probs = data['y_prob']
        p_list, r_list, f1_list = [], [], []
        for t in thresholds:
            p_bin = (probs >= t).astype(int)
            p_list.append(precision_score(y_test, p_bin, zero_division=0))
            r_list.append(recall_score(y_test, p_bin, zero_division=0))
            f1_list.append(f1_score(y_test, p_bin, zero_division=0))
            
        best_t_idx = np.argmax(f1_list)
        best_t = thresholds[best_t_idx]
        best_f1 = f1_list[best_t_idx]
        threshold_results[name] = (best_t, best_f1)
        
        axes[idx].plot(thresholds, p_list, label='Precision', color='#377eb8', linewidth=1.8)
        axes[idx].plot(thresholds, r_list, label='Recall', color='#e41a1c', linewidth=1.8)
        axes[idx].plot(thresholds, f1_list, label='F1-Score', color='#4daf4a', linewidth=2.2)
        axes[idx].axvline(x=0.5, color='black', linestyle=':', label='Default Cutoff (0.50)')
        axes[idx].axvline(x=best_t, color='purple', linestyle='--', label=f'Optimal F1 Cutoff ({best_t:.2f})')
        axes[idx].set_title(f"{name}: Metrics vs. Decision Threshold", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Probability Decision Threshold', fontsize=11)
        axes[idx].set_ylabel('Score', fontsize=11)
        axes[idx].legend(loc='lower left', fontsize=9)
        axes[idx].grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    p9 = os.path.join(RESULTS_DIR, '09_threshold_tuning.png')
    plt.savefig(p9)
    plt.close()
    print(f"-> Saved: {p9}")

    # Plot 10: Overfitting and Underfitting Comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(overfitting_data))
    width = 0.25
    
    rects1 = ax.bar(x - width, [d['Train ROC-AUC'] for d in overfitting_data], width, label='Train ROC-AUC', color='#4575b4', edgecolor='black')
    rects2 = ax.bar(x, [d['CV ROC-AUC (5-Fold)'] for d in overfitting_data], width, label='5-Fold CV ROC-AUC', color='#74add1', edgecolor='black')
    rects3 = ax.bar(x + width, [d['Test ROC-AUC'] for d in overfitting_data], width, label='Test ROC-AUC', color='#fdae61', edgecolor='black')
    
    ax.set_ylabel('ROC-AUC Score', fontsize=11)
    ax.set_title('Overfitting & Generalization Analysis (Train vs. CV vs. Test)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([d['Model'] for d in overfitting_data], fontsize=11, fontweight='bold')
    ax.set_ylim(0.5, 0.9)
    ax.legend(loc='lower right', frameon=True)
    
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height + 0.005, f'{height:.4f}', ha='center', va='bottom', fontsize=9)
            
    plt.tight_layout()
    p10 = os.path.join(RESULTS_DIR, '10_overfitting_analysis.png')
    plt.savefig(p10)
    plt.close()
    print(f"-> Saved: {p10}")

    # Executive Summary Console Output
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(df_results[['Model', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score', 'Test ROC-AUC']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("OVERFITTING / GENERALIZATION DIAGNOSTIC")
    print("=" * 80)
    print(df_overfitting[['Model', 'Train ROC-AUC', 'CV ROC-AUC (5-Fold)', 'Test ROC-AUC', 'AUC Generalization Gap (Train - Test)']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("OPTIMAL DECISION THRESHOLDS (MAXIMIZING F1-SCORE)")
    print("=" * 80)
    for name, (opt_t, opt_f1) in threshold_results.items():
        print(f"-> {name}: Optimal Threshold = {opt_t:.2f} (Yields F1 = {opt_f1:.4f} vs Default t=0.5)")
    print("=" * 80)


# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================
def main():
    print("=" * 80)
    print("PREDICTIVE ANALYTICS PROJECT: CREDIT DEFAULT PREDICTION")
    print("MODELS: LOGISTIC REGRESSION & LINEAR DISCRIMINANT ANALYSIS (LDA)")
    print("ACADEMIC SYLLABUS COMPLIANT IMPLEMENTATION")
    print("=" * 80)
    
    # 1. Ingestion
    df_raw = load_dataset()
    
    # 2. Data Cleaning & Renaming
    df_clean = clean_and_harmonize_data(df_raw)
    
    # 3. Domain Feature Engineering
    df_featured = engineer_features(df_clean)
    
    # 4. Exploratory Data Analysis
    perform_eda(df_featured)
    
    # 5. Data Splitting & Preprocessing
    X_train, X_test, y_train, y_test, preprocessor = prepare_data_and_pipeline(df_featured)
    
    # 6. Model Training & Hyperparameter Tuning
    models = train_and_tune_models(X_train, y_train, preprocessor)
    
    # 7. Model Evaluation & Comparison
    evaluate_and_compare(models, X_train, y_train, X_test, y_test)
    
    print("\n[SUCCESS] Project workflow completed cleanly. All artifacts saved to 'results/' directory.")


if __name__ == '__main__':
    main()
