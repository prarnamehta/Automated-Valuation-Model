# Automated-Valuation-Model
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
5. 

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

