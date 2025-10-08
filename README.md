# Winnipeg Residential Property Valuation Model

(a) Accessible description of the work for a non-technical client, not exceeding 200 words.

## Non-Technical Description
### Project Objective
This project presents an **Automation Valuation Model (AVM)** to estimate valuation of a residential property in Winnipeg using machine learning. The model is trained on historic property data from City of Winnipeg public data repository and predicts residential property’s value based on features like year built, number of rooms, neighborhood, and living area.

### What It does

The model analyzes over 200,000 residential property records to learn patterns for example, which Yneighbourhoods are more popular or how living area affects price. These learned patterns are then used to estimate a fair value of property when a user enters details.

### How it was built

The model is built using [**Python**](https://www.python.org/) and [**Jupyter Notebook**](https://jupyter.org/).  The data was cleaned, standardized, and filtered to focus on residential properties. Missing values were filled, and categorical data was encoded for modeling. Three algorithms—Random Forest, XGBoost, and Linear Regression—were tested with two scaling methods to find the best performer.

### Key Results

 The key influential features n predicting property value were,
 - **Total Living Area**
 - **Year Built**
 - **Neighborhood Area**
 - **Zoning**
 - **Garage and Basement Types**

(187 words)

(b) A more detailed description of the work with technical specifics justifying your modelling choices that does not exceed 500 words.

## Technical Description

A simple automation model is build to predict the assessed value of a residential property in Winnipeg. The Softwrae development workflow is as following:

**1. Data Collection**
The csv data was extracted from [Assessment dataset](https://data.winnipeg.ca/Assessment-Taxation-Corporate/Assessment-Parcels/d4mq-wa44/about_data)

### Data Preparation
The data went through following pre-processing steps,

-**Cleaning**: Removed empty/irrelevant columns and rows with corrupted values

-**Standardization**: Converted special characters ( e.g. '$',',') in numeric fields to proper numerical formats

-**Filtering**: Focused on only residential property types

-**Imputation**: Filled missing values using statistical methods

### Feature Engineering

•	**Low-cardinality categorical features** (≤ 20 unique values) were one-hot encoded.

•	**High-cardinality features** were frequency encoded.

•	Redundant or high-dimensional columns (e.g., Roll Number, Status, Property Class etc.) were dropped.

### Correlation & Feature Selection

A correlation matrix was used to identify features most related to Total Assessed Value. Features weith low or no correlation (e.g., GISID. Centroid Lon/Lat) were removed.

### Model Training
The dataset was split into *80% training* and *20% testing*
Three Machine learning algorithms were trained and assessed:

- **Random Forest Regressor**

- **XGBoost Regressor**

- **Linear Regression**

Each model was tested with two scaling techniques:
- **Min-Max Scaling**

- **Z-Score Standardization**

Feature scaling normalises the values after encoding categorical variables that helps machine learning algorithms to correctly work.

### Model Evaluation

Models were evaluated using:

- **R² (R-squared)** – How well the model explains price variation

- **MAPE** – Mean Absolute Percentage Error measures how far off the model’s predictions are, on average, as a percentage of the actual property values.

- **nRMSE** – Normalized Root Mean Squared Error measures how much the predictions deviate from the actual values, but it scales the error based on the range of property prices in the dataset.

Random Forest and XGBoost performed best, with high R² and low error rates.

### Visualizations

1. **Feature Improtance Chart** : Averaged from Random Forest and XGBoost showing top predictors: Total living area, Year Built, Neighbourhood Area, Zoning, Garage/ Basement
2. **Actual vs. Predicted Scatter Plots** : The clustering around the diagonal for XGBoost with Z-Score scaling indicating it was a good fit
3. **Residual Plots** : The residuals for Random Forest with min-max scaling was randomly distributed around zero indicating no bias
4. **Runtime & MSE charts** : Linear Regression has shortest runtime but low accuracy whereas XGBoost has high accuracy but long runtime

## Assumptions:
1. The assessed value is a reliable proxy for market value, hence used as target variable.
2. The residential properties are identified using Property Use Code
3. Missing values are imputed using median/mode
4. The models: [Random Forest and XGBoost are robust](http://www.sciencedirect.com/science/article/pii/S0264275124003299) and best used with dataset with numerical and categorical features. These algorithms also helps with feature importance that optimises the prediction model results.
5. Due to time constraints, the deployment of the prediction model was kept simple. 

## Limitations

While the model provides useful estimates, it does **not account for**:
- Interior upgrades or renovations
- Market volatility or economic shifts
- Unique property features not captured in the dataset

It is intended to display technical skills and not a replacement of professional appraisals

---

## Outputs

- `model_performance_comparison.csv` – Evaluation metrics for each model
- `feature_importance_comparison.csv` – Ranked feature importances
- `*.png` – Visualizations of predictions and residuals


## Acknowledgement:
1. http://www.sciencedirect.com/science/article/pii/S0264275124003299
2. Microsoft Co-Pilot was used to brush up deployment skills and knowledge, GitHub upload, deal with large datasets in Streamlite 

## Try the App (Deployment)

You can test the model using this interactive app:
[🔗 Automated Valuation App](https://automated-valuation-model.streamlit.app/)

To use it:

Download the property data from the link above.
Enter basic property details into the app to get an estimated value.

Or launch the full notebook here:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/prarnamehta/AVM/HEAD?filepath=Automated%20valuation%20model.ipynb)

Please Note: The firewall of an organisation might prevent from accessing the datafile. In that case, please view the pdf file of the same code to see the charts and results
