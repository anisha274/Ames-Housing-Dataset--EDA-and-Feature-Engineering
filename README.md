# Ames Housing Dataset – Exploratory Data Analysis & Feature Engineering

## 📌 Project Overview
This repository contains a comprehensive, production-grade data preprocessing, Exploratory Data Analysis (EDA), and Feature Engineering pipeline executed on the famous **Ames Housing Dataset** (compiled by Dean De Cock). The objective of this project is to take a raw, highly skewed, and structurally messy real estate dataset containing **2,930 observations and 82 features** and transform it into a leakage-free, fully optimized matrix ready for supervised machine learning regression algorithms.

This work was completed as part of the **Big Data Lab Assignment - 1**.
Kaggle link🔗 : https://www.kaggle.com/code/ummemehajabinanisha/ames-housing-dataset-eda-feature-engineering 
---

## 🛠️ Key Pipeline Accomplishments

* **Structural Imputation:** Successfully resolved all 27 missing-value features without throwing away data. Used domain logic to classify missingness into **Structural Absence** (e.g., mapping `NaN` pools or garages to `'None'` or `0`) versus **Random Missingness** (imputed using robust median/mode strategies).
* **Outlier & Skewness Mitigation:** Identified heavy right-skewness in key metrics (`SalePrice`, `GrLivArea`, `LotArea`). Implemented mathematical transformations (**Log1p** and **Yeo-Johnson Power Transformations**) lowering the target skewness from `1.74` to near `0` for enhanced regression stability.
* **Domain-Driven Feature Engineering:** Appended 5 highly correlated engineered features derived from real estate business logic:
    * `TotalLivingArea` ($r = 0.790$): Combining above-ground and basement square footage.
    * `HouseAge` ($r = -0.559$): Calculating active depreciation from year built to year sold.
    * `TotalBathrooms` ($r = 0.636$): Weighted accumulation of full and half baths.
    * `TotalPorchArea` ($r = 0.384$): Aggregating outdoor living utility spaces.
    * `RemodelingAge` ($r = -0.535$): Quantifying structural freshness since last renovation.
* **Leakage-Free Preprocessing:** Built a secure `ColumnTransformer` execution pipeline utilizing an `80/20 Train-Test Split`. Data scaling (`StandardScaler`), Ordinal Mapping, and One-Hot Encoding were fitted **strictly** on the training split to eliminate data leakage.

---

## 📊 Core Analytical Insights Found
1. **Exponential Quality Scaling:** Real estate value scales exponentially rather than linearly with material finishes; moving from quality score 8 $\rightarrow$ 9 yields a $4\times$ larger marginal price leap than moving from 5 $\rightarrow$ 6.
2. **High Multicollinearity:** Garage features (`GarageYrBlt`, `GarageCars`, `GarageArea`) exhibit severe internal collinearity, indicating that models require regularization (like LASSO) to compress coefficients safely.
3. **Location Superiority:** Median property pricing fluctuates by more than $3.4\times$ strictly across different Ames neighborhoods (e.g., `NoRidge` vs `MeadowV`), marking location as a primary valuation anchor.

---

## 📈 Baseline Model Results
Following the completion of the preprocessing pipeline, baseline evaluations yielded exceptional predictive coverage:
* **LASSO Regression:** Achieved an **$R^2$ Score of 0.8893** with a Mean Absolute Error (**MAE**) of **$17,030.87**. LASSO successfully dropped 72 redundant features using L1 regularization.
* **Random Forest Regressor:** Achieved an **$R^2$ Score of 0.9183** with a Mean Absolute Error (**MAE**) of **$15,717.99**.

---

## 📁 Repository Structure
* `Ames_Housing_Assignment.ipynb` : The main Jupyter notebook holding the complete visual matrix, code blocks, and markdown annotations.
* `README.md` : Project documentation summary.

## 🧮 Technology Stack
* **Languages:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-Learn, SciPy, Matplotlib, Seaborn
* **Environments:** Google Colab / Kaggle Notebooks
