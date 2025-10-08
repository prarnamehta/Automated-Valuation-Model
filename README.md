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



(b) A more detailed description of the work with technical specifics justifying your modelling choices that does not exceed 500 words.

## Technical Description

A simple automation model is build to predict the assessed value of a residential property in Winnipeg. The Softwrae development workflow is as following:

**1. Data Collection**
The csv data was extracted from [Assessment dataset](https://data.winnipeg.ca/Assessment-Taxation-Corporate/Assessment-Parcels/d4mq-wa44/about_data)

#### 🧹 Data Preparation
The data went through following pre-processing steps,

-**Cleaning**: Removed empty/irrelevant columns and rows with corrupted values

-**Standardization**: Converted special characters ( e.g. '$',',') in numeric fields to proper numerical formats

-**Filtering**: Focused on only residential property types

-**Imputation**: Filled missing values using statistical methods

-**Encoding**: Categorical features were converted to numerics

#### Model Training
The dataset was split into *80% training* and *20% testing*
Three Machine learning algorithms were trained and assessed:

- **Random Forest Regressor**

- **XGBoost Regressor**

- **Linear Regression**

Each model was tested with two scaling techniques:
- **Min-Max Scaling**

- **Z-Score Standardization**

Feature scaling normalises the values after encoding categorical variables that helps machine learning algorithms to correctly work.

## 📈 Model Evaluation

Models were evaluated using:

- **R² (R-squared)** – How well the model explains price variation

- **MAPE** – Mean Absolute Percentage Error measures how far off the model’s predictions are, on average, as a percentage of the actual property values.

- **nRMSE** – Normalized Root Mean Squared Error measures how much the predictions deviate from the actual values, but it scales the error based on the range of property prices in the dataset.


**2. Data Pre-Processing**

•	Drops empty and irrelevant columns.

•	Cleans numeric columns with special characters (e.g., $, ,).

•	Handles missing values using median for numerical columns and mode or "Missing" for categorical columns.

•	Filters for residential properties using Property Use Code.

**3. Feature Engineering**

•	Drops high-cardinality categorical columns ( columns with more than 500 categories)

•	Applied: 
o	One-hot encoding for low-cardinality categorical features.
o	Frequency encoding for high-cardinality ones.

•	Feature Scaling:
o	MinMaxScaler
o	StandardScaler
Both of the methods were compared to identify the best performing model

**4. Modeling**
•	Trains two models: RandomForestRegressor, and XGBRegressor.
•	Evaluates each with: 
o	Metrics: R², MAPE, nRMSE.
•	Saves: 
o	Performance metrics (model_performance_comparison.csv)
o	Feature importances (feature_importance_comparison.csv)
o	Visualizations (scatter plots and residuals)



### Assumptions:
1. The dataset extracted from the above mentioned website was a large file. I used a random subset of the file to tackle the problem.
2. The assignment asked for prediction model for residential properties, hence an extracted subset was used to test
3. I used 'Total assessed value' as my dependent variable
4. The models: Random Forest and XGBoost (Reference 1) are robust and best used with dataset with numerical and categorical features. These algorithms also helps with feature importance that optimises the prediction model results.
5. Due to time constraints and lack of expert knowledge, the deployment of the prediction model was kept simple. 

## ⚠️ Limitations

While the model provides useful estimates, it does **not account for**:
- Interior upgrades or renovations
- Market volatility or economic shifts
- Unique property features not captured in the dataset

It is intended to display technical skills and not a replacement of professional appraisals

---

## 📂 Outputs

- `model_performance_comparison.csv` – Evaluation metrics for each model
- `feature_importance_comparison.csv` – Ranked feature importances
- `*.png` – Visualizations of predictions and residuals


### Acknowledgement:
1. http://www.sciencedirect.com/science/article/pii/S0264275124003299
2. Microsoft Co-Pilot to deploy the prediction model, to brush up skills and knowledge.


### Try the App (Deployment)

You can test the model using this interactive app:
[🔗 Automated Valuation App](https://automated-valuation-model.streamlit.app/)

To use it:

Download the property data from the link above.
Enter basic property details into the app to get an estimated value.

Or launch the full notebook here:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/prarnamehta/AVM/HEAD?filepath=Automated%20valuation%20model.ipynb)

Please Note: The firewall of an organisation might prevent from accessing the datafile.
