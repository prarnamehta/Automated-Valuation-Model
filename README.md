# Automated-Valuation-Model

(a) Accessible description of the work for a non-technical client, not exceeding 200 words.

## Non-Technical Description
### Project Description
The model, **Automation Valuation Model (AVM)** developed is to estimate valuation of a residential property in Winnipeg based on property's features. An automated valuation models is a tool based on mathematical and statistical modelling that uses property features like year built, number of rooms, living square footage etc to assess property's value. The model is trained using historic data extracted from City of Winnipeg public data repository. The goal is create a simple estimaion model based on key features.

### What It does
The model predicts the valuation of a residential property using information like
 
 * Year Built

 * Number of rooms

 * Neighbourhood Area

 * Living Area

### How it works

The model has reviewed 203558 assessed value of residential properties in Winnipeg. It learns different patterns in the historic data for example, which Year built had greater value, which area was more popular, average living area value. These learned patterns are then used to estimate a fair value of a property when a user enters details.

### How it was built

The model is built using Python, a programming language for data science, using Jupyter Notebook that is an interactive platform to clean and explore data.  

Due to time constraints, the model developed is extremely simple but supported by strong research. This automation is based on historic data that might not include all factors to predict the valuation. 

The first step to build the model was to preprocess the data

(b) A more detailed description of the work with technical specifics justifying your modelling choices that does not exceed 500 words.

## Technical Description

A simple automation model is build to predict the assessed value of a residential property in Winnipeg. The Softwrae development workflow is as following:

**1. Data Collection**
The csv data was extracted from [Assessment dataset](https://data.winnipeg.ca/Assessment-Taxation-Corporate/Assessment-Parcels/d4mq-wa44/about_data)

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

