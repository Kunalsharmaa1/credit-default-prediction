# Credit Default Prediction Using Logistic Regression and Linear Discriminant Analysis (LDA)

An academic Predictive Analytics project strictly adhering to syllabus requirements, evaluating **Logistic Regression** and **Linear Discriminant Analysis (LDA)** on the **UCI Default of Credit Card Clients Dataset (ID: 350)**.

---

## 1. Project Overview & Academic Scope

This project develops an end-to-end, leak-free predictive analytics pipeline to assess credit default risk among credit-card holders. In accordance with university Predictive Analytics course constraints:
- **Eligible Machine Learning Models**: Strictly **Logistic Regression** and **Linear Discriminant Analysis (LDA)**.
- **Excluded Architectures**: Tree ensembles (Random Forest, XGBoost, LightGBM), Support Vector Machines, Neural Networks, and Deep Learning models are intentionally omitted.
- **Workflow Components**:
  - Automated data ingestion from the UCI Machine Learning Repository (`ucimlrepo`).
  - Exploratory Data Analysis (EDA) and publication-ready diagnostic charts.
  - Domain-specific financial feature engineering.
  - Leak-free preprocessing using scikit-learn `Pipeline` and `ColumnTransformer`.
  - 5-Fold Stratified Cross-Validation and Hyperparameter Grid Search.
  - Comprehensive evaluation: Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, and Confusion Matrices.
  - Overfitting/Underfitting diagnostics and Decision Threshold Optimization.

---

## 2. Dataset Description & Demographic Context

- **Source**: UCI Machine Learning Repository — [Dataset ID: 350 (Default of Credit Card Clients)](https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients)
- **Observations**: 30,000 raw customer records; **29,965 unique profiles** after removing 35 exact duplicates.
- **Attributes**: 23 predictive features covering credit limits, client demographics, six-month payment histories (April to September 2005), statement billing amounts, and previous repayment amounts.
- **Target Variable (`DEFAULT`)**: Binary indicator (1 = default on payment next month, 0 = paid on time).
  - Non-Default ($0$): 23,335 records (**77.87%**)
  - Default ($1$): 6,630 records (**22.13%**)

### Critical Context: Taiwan (2005) vs. Indian Credit Market

> [!WARNING]
> **Geographic & Regulatory Discrepancy**:
> This dataset records revolving credit card clients in **Taiwan between April and September 2005**, during the expansion of consumer credit preceding Taiwan's 2006 credit card debt crisis.
> 
> **Why it does not directly represent Indian loan applicants:**
> 1. **Credit Product Scope**: Revolving unsecured credit card debt carries fundamentally different default dynamics than Indian retail credit products (secured home loans, auto loans, gold loans, or agricultural Kisan Credit Cards).
> 2. **Credit Bureau Infrastructure**: In India, underwriting heavily depends on formal credit scores (CIBIL / Experian / CRIF High Mark, 300–900 scale) and RBI Account Aggregator financial transaction histories, features not present in this dataset.
> 3. **Regulatory Framework**: The Reserve Bank of India (RBI) enforces strict NPA recognition (90-day overdue norms) and priority sector lending requirements, distinct from Taiwan's 2005 Financial Supervisory Commission landscape.
> 4. **Macroeconomic Realities**: Taiwanese monetary values are in New Taiwan Dollars (NT$), reflecting 2005 East Asian urban purchasing power, which differs significantly from Indian household income structures.

---

## 3. Project Architecture

```
credit_default_prediction/
├── .gitignore               # Excludes bytecode, venvs, and temporary files
├── requirements.txt         # Pinned project dependencies
├── main.py                  # End-to-end executable workflow
├── README.md                # Comprehensive project documentation & viva guide
├── data/                    # Local cache directory for UCI dataset
│   └── default_of_credit_card_clients.csv
└── results/                 # Publication-ready figures and metrics
    ├── 01_target_distribution.png
    ├── 02_demographic_default_rates.png
    ├── 03_repayment_status_impact.png
    ├── 04_credit_limit_distribution.png
    ├── 05_correlation_heatmap.png
    ├── 06_confusion_matrices.png
    ├── 07_roc_curves.png
    ├── 08_precision_recall_curves.png
    ├── 09_threshold_tuning.png
    ├── 10_overfitting_analysis.png
    ├── model_comparison.csv
    └── model_comparison.md
```

---

## 4. Methodology & Engineering Workflow

### A. Data Cleaning & Schema Harmonization
1. **Semantic Renaming**: UCI column designations `X1`–`X23` and `Y` were mapped to standard credit terminology (`LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, `PAY_0`–`PAY_6`, `BILL_AMT1`–`BILL_AMT6`, `PAY_AMT1`–`PAY_AMT6`, and `DEFAULT`).
2. **De-duplication**: 35 identical records were pruned to prevent artificial data inflation.
3. **Category Cleaning**: Undocumented levels in `EDUCATION` ($0, 4, 5, 6 \rightarrow 4$ "Others") and `MARRIAGE` ($0, 3 \rightarrow 3$ "Others") were harmonized in accordance with econometric literature standards.

### B. Domain Feature Engineering
Five domain-specific features were engineered based on retail credit risk principles:
- **`PAY_MAX`**: $\max(\text{PAY\_0}, \dots, \text{PAY\_6})$ — Identifies the peak delinquency severity over the past six months.
- **`BILL_AMT_AVG`**: Average statement balance over the six-month observation window.
- **`PAY_AMT_AVG`**: Average monthly payment made over the six-month observation window.
- **`UTILIZATION_1`**: $\text{BILL\_AMT1} / (\text{LIMIT\_BAL} + 1)$ — Credit utilization in the latest billing period (clamped to $[-1.0, 5.0]$).
- **`PAY_TO_BILL_RATIO_1`**: $\text{PAY\_AMT1} / (|\text{BILL\_AMT2}| + 1)$ — Repayment compliance relative to the previous billing statement.

### C. Leak-Free Preprocessing & Pipeline Architecture
- **Stratified Split**: 80% Training ($N = 23,972$), 20% Untouched Testing ($N = 5,993$), stratified by `DEFAULT` to preserve the 22.13% target distribution across both partitions.
- **Encapsulated Preprocessor**: Built using scikit-learn `ColumnTransformer`:
  - Continuous numerical features (25 features) standardized using `StandardScaler()`.
  - Nominal categorical features (`SEX`, `EDUCATION`, `MARRIAGE`) encoded via `OneHotEncoder(drop='first', sparse_output=False)`.
  - Scalers and encoders are **fitted strictly on training data** and applied to test data without refitting, preventing data leakage.

---

## 5. Actual Experimental Results

Both models were trained on the identical training split and evaluated on the identical, untouched test split ($N = 5,993$). All metrics reported below are genuine empirical outputs from `main.py`.

### Comparative Performance Summary Table

| Metric | Logistic Regression | Linear Discriminant Analysis (LDA) |
| :--- | :---: | :---: |
| **Best Hyperparameters** | `{'C': 10.0, 'class_weight': 'balanced', 'solver': 'lbfgs'}` | `{'shrinkage': 'auto', 'solver': 'lsqr'}` |
| **5-Fold CV ROC-AUC** | **0.7409** | 0.7342 |
| **Test ROC-AUC** | **0.7339** | 0.7257 |
| **Test Accuracy** | 0.7245 (72.45%) | **0.8106 (81.06%)** |
| **Test Precision (Default)** | 0.4194 (41.94%) | **0.6792 (67.92%)** |
| **Test Recall / Sensitivity** | **0.6373 (63.73%)** | 0.2730 (27.30%) |
| **Test Specificity** | 0.7493 (74.93%) | **0.9634 (96.34%)** |
| **Test F1-Score** | **0.5058** | 0.3895 |
| **True Negatives (TN)** | 3,497 | 4,496 |
| **False Positives (FP)** | 1,170 | 171 |
| **False Negatives (FN)** | **481** | 964 |
| **True Positives (TP)** | **845** | 362 |

---

## 6. In-Depth Predictive Analytics Insights

### A. The Accuracy Paradox in Imbalanced Credit Portfolios
- The test set contains 4,667 non-defaulters and 1,326 defaulters (a 77.87% / 22.13% split).
- A naive baseline classifier that predicts "No Default" for every client achieves **77.87% accuracy**, but possesses a Recall of **0.0%**, missing all 1,326 defaults.
- While LDA achieves a superficially high accuracy of **81.06%**, it misclassifies **964 out of 1,326 defaulting clients** as non-defaulters (Recall: **27.30%**).
- Logistic Regression, configured with `class_weight='balanced'`, prioritizes minority-class detection, achieving **63.73% Recall** and catching **845 defaulters**, slashing False Negatives by **50.1%** (from 964 down to 481).

### B. Business Cost Trade-Off: False Negatives vs. False Positives
In financial credit risk management, errors are fundamentally asymmetric:
- **False Negative (FN)**: Approving a borrower who subsequently defaults. The bank loses the entire outstanding principal amount (credit loss / write-off).
- **False Positive (FP)**: Denying credit or placing holds on a creditworthy borrower. The bank loses transactional interchange fees and causes customer friction.
- **Cost Ratio**: In retail lending, an FN typically costs $5\times$ to $10\times$ more than an FP. Logistic Regression's higher sensitivity (0.6373) significantly reduces expected portfolio loss compared to default-threshold LDA.

### C. Decision Threshold Optimization
- Default classification applies a rigid probability cutoff at $t = 0.50$.
- By plotting Precision, Recall, and F1 across thresholds $t \in [0.10, 0.90]$ (see `09_threshold_tuning.png`):
  - **Logistic Regression**: Achieves its optimal F1-score of **0.5193** at threshold $t = 0.57$.
  - **LDA**: Achieves its optimal F1-score of **0.5170** at threshold $t = 0.27$. Lowering LDA's threshold from $0.50$ to $0.27$ dramatically improves its default detection capability.

### D. Overfitting and Generalization Diagnostic

| Model | Train ROC-AUC | 5-Fold CV ROC-AUC | Test ROC-AUC | Generalization Gap (Train - Test) | Overfitting Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.7437 | 0.7409 | 0.7339 | **0.0098** | Negligible gap (< 0.010); no overfitting. |
| **Linear Discriminant Analysis** | 0.7365 | 0.7342 | 0.7257 | **0.0108** | Negligible gap (~ 0.011); robust generalization. |

Both models display minimal degradation between training and test sets, proving that L2 regularization in Logistic Regression and Ledoit-Wolf automatic covariance shrinkage (`shrinkage='auto'`) in LDA prevented over-parameterization.

---

## 7. Generated Visualizations

All high-resolution diagnostic plots are saved in the `results/` folder:

1. **`01_target_distribution.png`**: Class imbalance bar chart displaying frequencies and percentages.
2. **`02_demographic_default_rates.png`**: 4-panel segment analysis across Gender, Education, Marital Status, and Age brackets.
3. **`03_repayment_status_impact.png`**: Default rate progression across September repayment delinquency statuses (`PAY_0`).
4. **`04_credit_limit_distribution.png`**: Box plot and mean indicators showing credit limit disparities by default status.
5. **`05_correlation_heatmap.png`**: Correlation matrix highlighting correlations between payment statuses and the default target.
6. **`06_confusion_matrices.png`**: Side-by-side normalized and raw confusion matrices for both models.
7. **`07_roc_curves.png`**: Receiver Operating Characteristic curves with AUC scores.
8. **`08_precision_recall_curves.png`**: Precision-Recall curves illustrating trade-offs against the 0.221 baseline.
9. **`09_threshold_tuning.png`**: Precision, Recall, and F1 trajectories across the full probability spectrum.
10. **`10_overfitting_analysis.png`**: Train vs. CV vs. Test ROC-AUC bar comparison confirming absence of overfitting.

---

## 8. Setup & Execution Guide

### Prerequisites
- Python 3.9+ (tested on Python 3.14.2)
- Git

### Installation
Clone or navigate to the project directory:
```bash
cd credit_default_prediction
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

### Running the Workflow
Execute the end-to-end pipeline:
```bash
python main.py
```

The script will automatically check local data cache, perform cleaning and feature engineering, output all EDA figures, execute 5-fold cross-validation grid searches, print evaluation metrics to the console, and generate comparison tables in `results/`.

---

## 9. Viva Preparation: Key Academic Questions & Answers

### Q1: What is the fundamental difference in mathematical formulation between Logistic Regression and Linear Discriminant Analysis (LDA)?
**Answer**:
- **Logistic Regression** is a **discriminative** probabilistic model. It directly models the posterior class probability $P(Y=1 \mid X)$ using the logistic sigmoid function:
  $$P(Y=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \boldsymbol{\beta}^T X)}}$$
  Parameters are estimated via **Maximum Likelihood Estimation (MLE)** without assuming any specific probability distribution for the feature vector $X$.
- **Linear Discriminant Analysis (LDA)** is a **generative** model. It models the class-conditional feature distributions $P(X \mid Y=k)$ and uses Bayes' Theorem to derive $P(Y=k \mid X)$. LDA makes two strong distributional assumptions:
  1. Features within each class follow a multivariate normal distribution: $X \mid Y=k \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma})$.
  2. All classes share an identical covariance matrix ($\boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}$).
  Parameters are computed analytically via sample means and pooled covariance matrices rather than iterative numerical optimization.

### Q2: Why is accuracy an unreliable metric for evaluating credit default prediction models?
**Answer**:
Because credit default datasets suffer from severe class imbalance (~78% non-defaulters vs. ~22% defaulters). A trivial majority-class classifier that blindly predicts "non-default" for every applicant achieves 77.87% accuracy while detecting zero defaults (Recall = 0%). In banking, missing a defaulting borrower (False Negative) leads to charge-offs and bad debts, which are far more costly than False Positives. Therefore, **ROC-AUC**, **Recall**, and **F1-Score** are the appropriate evaluation criteria.

### Q3: How did you ensure zero data leakage during data preprocessing and feature scaling?
**Answer**:
All transformations that compute dataset statistics (such as the mean and standard deviation for `StandardScaler`, or categorical frequencies for `OneHotEncoder`) were encapsulated inside a scikit-learn `Pipeline` and `ColumnTransformer`. The transformer was fitted exclusively on the training split (`X_train`) and subsequently used to transform the test split (`X_test`). No information from the test split was ever exposed during training or preprocessing.

### Q4: Why did LDA achieve high accuracy but low recall at the standard 0.5 threshold?
**Answer**:
LDA's classification rule assigns an observation to the class that maximizes the linear discriminant function:
$$\delta_k(x) = x^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k - \frac{1}{2} \boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \log(\pi_k)$$
where $\pi_k$ represents the prior probability of class $k$. Because $\pi_0 \approx 0.78$ and $\pi_1 \approx 0.22$, the prior term heavily penalizes predictions for the minority class, requiring overwhelming evidence before predicting default. Lowering the decision threshold (e.g., to $t = 0.27$) compensates for the unbalanced priors and increases recall to viable levels.

### Q5: How was overfitting assessed and controlled?
**Answer**:
1. **Regularization**: L2 regularization was applied to Logistic Regression (tuned over $C \in [0.01, 10.0]$), and automatic covariance shrinkage (Ledoit-Wolf) was applied to LDA.
2. **Evaluation Across Splits**: We systematically compared Train ROC-AUC, 5-Fold Cross-Validation ROC-AUC, and untouched Test ROC-AUC. The generalization gap was under 0.011 for both models, demonstrating that neither model suffered from high variance or overfitting.

---

## 10. Ethical Considerations & Project Limitations

1. **Demographic Proxies and Algorithmic Fairness**: Attributes such as `SEX`, `EDUCATION`, and `MARRIAGE` must be evaluated under fair lending guidelines (such as the Equal Credit Opportunity Act or Indian banking non-discrimination norms) to ensure that predictive scoring does not perpetuate systemic bias against protected groups.
2. **Lack of Bureau Data**: The dataset lacks external bureau scores (e.g., CIBIL/Experian), debt-to-income (DTI) ratios, and asset collateral information, which are primary underwriting determinants in contemporary commercial banking.
3. **Temporal Invariance**: The data represents a static six-month window in 2005. Credit behaviors shift dynamically across macroeconomic cycles, interest rate changes, and inflation fluctuations, requiring continuous model monitoring and drift re-estimation in production.
