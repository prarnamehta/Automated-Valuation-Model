#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data analysis
# 
# ## Data Loading
# 
# Installing data analysis packages/libraries and loading csv datafile using url

# In[1]:


import pandas as pd  # For data loading, manipulation, and analysis using DataFrames
import seaborn as sns  # For advanced statistical data visualization (e.g., heatmaps, bar plots)
import numpy as np  # For numerical operations, arrays, and mathematical functions
import time  # To measure model training and prediction runtimes
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For feature scaling
from sklearn.ensemble import RandomForestRegressor  # Ensemble model for regression using decision trees
from sklearn.linear_model import LinearRegression  # Simple linear regression model for baseline comparison
from xgboost import XGBRegressor  # Gradient boosting model for high-performance regression tasks
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error  
# For evaluating model performance using metrics like R², MAPE, and RMSE
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables
import warnings  # To suppress warnings during model training and data processing
from datetime import datetime # to get the current year
warnings.filterwarnings("ignore")


url = "https://data.winnipeg.ca/api/views/d4mq-wa44/rows.csv?accessType=DOWNLOAD"
ds = pd.read_csv(url) # reading files directy from the website
ds


# In[2]:


# displaying summary of the data
ds.info()


# # Cleaning
# 
# Dropping columns with 0 non-null values

# In[3]:


# Identify columns with 0 non-null values
empty_columns = ds.columns[ds.isnull().all()].tolist()

# Drop all empty columns except the one to keep
columns_to_drop = [col for col in empty_columns ] #if col != column_to_keep
ds = ds.drop(columns=columns_to_drop)


# ## Descriptive statistics for numerical columns
# ### Comments
# 
# Identified few numerical columns that were of data type 'object'. 
# They have some speical characters like Dollar sign, comma etc that needs to be stripped.
# 
# After scrubbing, the descriptive statistics of numerical columns is as following,
# 
# Interpretations:
#  - there are missing values as the count are not same for all the columns, that might need imputing
#  - some of them are empty columns that needs to be dropped
#  - Due to large gaps in min and max values for columns like Total Living Area, Rooms etc. data might require normalisation for modelling purposes
#  - there are some irrelevant columns that are identifiers but not relevant for modelling like Roll number, Street number etc.

# In[4]:


# Numeric columns with special characters that need cleaning
columns_to_clean = ['Street Number','Total Living Area','Total Assessed Value','Assessed Land Area','Water Frontage Measurement','Sewer Frontage Measurement',
                    'GISID','Roll Number','Dwelling Units','Current Assessment Year']

for col in columns_to_clean:
    ds[col] = pd.to_numeric(ds[col].replace('[\$,]', '', regex=True), errors='coerce')
ds.describe()


# ## Comments:
# 
# While going through the dataset, two rows were identified with arbitary values, '0107376627423 49.80160863413996' in all the columns, so they were dropped.

# In[5]:


# Changing the data type of 'Roll Number' to object in order to remove the long random roll number
ds['Roll Number'] = ds['Roll Number'].astype(object)

# List of record IDs to drop
roll_nos_to_drop = ['0107376627423 49.80160863413996', '9257376']

# Drop rows where record_id is in the list
ds = ds[~ds['Roll Number'].isin(roll_nos_to_drop)]


# ## Descriptive statistics of categorical columns
# 
# Interpretations:
# - There are some high cardinal columsn like Street name, unit number, Full address, Geometry that might increase complexity of model and can be dropped
# - There are low cardinal columns that can be used to building type, Neighbourhood Area etc

# In[6]:


# Generate descriptive statistics for categorical columns
categorical_stats = ds.describe(include='object')
categorical_stats


# # Handling missing data
# 
# Deleting rows can lead to loss of information in large dataset, hence for this project imputation is used to fill missing values. 
# 
# Numerical variables like Total Living Area, Rooms, Water Frontage Measurement, Sewer Frontage Measurement, Total Assessed Value were replaced by Median
# 
# Categorical Variable like Street Type, Building type, Basement, Basement Finish, Year Built were replaced by mode or by introducing new category "Missing".
# 
# 

# In[7]:


# calculating number of null values in each column
ds.isnull().sum()


# In[8]:


# Drop the first unnamed column if it's just an index
if ds.columns[0].startswith('Unnamed'):
    ds = ds.iloc[:, 1:]

# Summary of missing values before imputation
missing_before = ds.isnull().sum()

# Separate numerical and categorical columns
numerical_cols = ds.select_dtypes(include=['number']).columns
categorical_cols = ds.select_dtypes(include=['object']).columns

# Handle missing values
# For numerical columns: median imputation
for col in numerical_cols:
    median_value = ds[col].median()
    ds[col].fillna(median_value, inplace=True)

# For categorical columns: mode imputation or 'Missing' category
for col in categorical_cols:
    mode_value = ds[col].mode()
    if not mode_value.empty:
        ds[col].fillna(mode_value[0], inplace=True)
    else:
        ds[col].fillna('Missing', inplace=True)

# Summary of missing values after imputation
missing_after = ds.isnull().sum()
missing_after


# # Comments:
# 
# The project requires prediction model for RESIDENTIAL PROPERTIES. So the column 'Property Use Code' is used to identify residential properties. 

# In[9]:


# List all unique categories in the 'Property Use Code' column to identify residential properties only
unique_categories = ds['Property Use Code'].dropna().unique().tolist()
unique_categories


# In[10]:


# Categories to filter
categories_to_keep = ['RESSD - DETACHED SINGLE DWELLING', 'RESMB - RESIDENTIAL MULTIPLE BUILDINGS',
                     'RESSS - SIDE BY SIDE','RESSS - SIDE BY SIDE','RESMH - MOBILE HOME',
                      'RESRM - ROOMING HOUSE','RESDU - DUPLEX','RESTR - TRIPLEX','RESRH - ROW HOUSING',
                      'RESMC - MULTIFAMILY CONVERSION','RESGC - RESIDENTIAL GROUP CARE',
                      'RESOT - RESIDENTIAL OUTBUILDING','RESSU - RESIDENTIAL SECONDARY UNIT',
                      'RESMA - MULTIPLE ATTACHED UNITS','RESMU - RESIDENTIAL MULTIPLE USE',
                      'RESAM - APARTMENTS MULTIPLE USE','RESAP - APARTMENTS','CNRES - CONDO RESIDENTIAL'
                     ]

# Filter the DataFrame to include only rows with specified residential type
ds = ds[ds['Property Use Code'].isin(categories_to_keep)]


# # Correlation analysis
# 
# The next step is to find relationship between target variable i.e. Total Assessed Value and other features with the help of correlation co-efficients. 
# - The features with values closer to +1 or -1 indicates stronger relationship with target variable. For example, Dwelling units
# - However any value closer to 0 indicates no or weak relationship for example Current Assessment Year, Centroid Lat, Roll Number, Centroid Lon and can be dropped.
# - Variables like Total Living area, Year Built, Rooms needs further investigation as they are key features for the prediction model

# In[11]:


# Select only numerical features for correlation analysis
numerical_dataset = ds.select_dtypes(include=['number'])
correlation=numerical_dataset.corr()
print(correlation['Total Assessed Value'].sort_values(ascending=False),'\n')


# # Comments:
# 
# Identifying object, integer, float type columns and assigning them to different subsets

# In[12]:


# identifying object type columns
obj = (ds.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

# identifying integer type columns
int_ = (ds.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

# identifying float type columns
fl = (ds.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))


# ### Comments:
#  - As per correlation analysis, dropping column 'Current Assessment Year', 'GISD','Street number', 'Centroid Lon', 'Centroid Lat'.
#  - Features with a large number of unique categories can be challenging to handle in machine learning models, as they might lead to high-dimensional data.The columns with more than 500 categories may not contribute to AVM. Hence they can be dropped. 
#  - Also columns with 1 category were looked into manually in excel file. They were just singular value against a property record which would play no role in prediction modelling, so they are dropped.
#  - Columns like Assessed Value, status, Property Class 1, 2,3,4,5 were dropped as we have a condensed column displaying the same values.

# In[13]:


Cols_to_drop = ['Full Address','Geometry','Detail URL', 'Assessment Date',
                'Assessed Value 1','Assessed Value 2','Assessed Value 3','Assessed Value 4',
                'Assessed Value 5','Property Class 1','Property Class 2','Property Class 3',
                'Property Class 4','Property Class 5','Status 1','Status 2','Status 3','Status 4',
                'Status 5','Unit Number','Street Name','Street Suffix','Dwelling Units',
                'Multiple Residences','Property Influences','Number Floors (Condo)', 'Street Number',
                'Current Assessment Year','GISID','Centroid Lon','Centroid Lat','Roll Number']
ds = ds.drop(columns=Cols_to_drop)


# ### Comments: 
# identifying number of unique categores each object type columns have.

# In[15]:


# identifying object type columns
obj = (ds.dtypes == 'object')
object_cols = list(obj[obj].index)


unique_values = []
for col in object_cols:
    unique_values.append(ds[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)


# In[16]:


category_counts = ds[object_cols].nunique()
category_counts


# In[17]:


# Select only numerical features for correlation analysis
numerical_dataset = ds.select_dtypes(include=['number'])

plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)


# ### Comments:
# 
# Variables like Total Living area, Year Built, Rooms were further investigation as they are key features for the prediction model
# 
# From all three scatterplots, one can analyse the following,
# - there a outliers that can distort the prediction models
# - there can be non-linear relationships which is well covered in random forests and xgboost
# - Histogram and boxplot can be created to understand further about the distributions and skewness

# In[18]:


sns.scatterplot(x=ds['Total Living Area'], y=ds['Total Assessed Value'])


# In[19]:


sns.scatterplot(x=ds['Rooms'], y=ds['Total Assessed Value'])


# In[20]:


sns.scatterplot(x=ds['Year Built'], y=ds['Total Assessed Value'])


# ### Checking for Data quality issues
# 
# - Total Living Area has maximum value as 11197 which significantly higher than 75th percentile indicating outliers. There can be records of properties that are large estates or multi-unit buildings. These large values can skew model training
# - The maximum date is 2025 that is the current year, some records may indicate future date, or data error. these records would need validation
# - Rooms has maximum value as 29 which might indicate mansions

# In[21]:


ds[['Total Living Area', 'Year Built', 'Rooms']].describe()


# ### Comments
# Histograms and Boxplots is used to further understand outliers and skewness in distribution for these features.
# 
# - **Total Living Area**
#     - Histogram:  The distribution is right skewed indicating most properties having living area between 500 to 2000 sq feet
#     - Box plot: The plot confirms peak of histogram and depicts outliers for Living area more than 2500 sq feet
#     The outliers can be capped at 99th percentile and with log transformation to normalise the distribution
# 
# - **Year Built**
#      - Histogram: this distribution is multimodal with distinct peaks early 1900, 1940 - 1960, 1980. Few builts before 1900 indicating outliers
#      - Box plot: The plot confirms outliers before 1900 which might have to handled
#     A derieved feature 'Age of Property' can be used instead to deal with outliers, and multimodal 
#      
# 
# - **Rooms**
#      - Histogram: The distribution is also right skewed indicating most properties has 5-10 rooms
#      - Box plot: The plot indicates outliers after 10 indicating few luxury properties
#     The outliers can be capped at 99th percentile 
# 

# In[22]:


features = ['Total Living Area', 'Year Built', 'Rooms']

for col in features:
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(ds[col].dropna(), kde=True, bins=50)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=ds[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)

    plt.tight_layout()
    plt.show()


# ### Dealing with outliers

# In[23]:


cap = ds['Total Living Area'].quantile(0.99)
ds['Total Living Area'] = ds['Total Living Area'].clip(upper=cap)


# In[24]:


ds['Log_LivingArea'] = np.log1p(ds['Total Living Area'])


# In[25]:


cap = ds['Rooms'].quantile(0.99)
ds['Rooms'] = ds['Rooms'].clip(upper=cap)


# ### Feature Engineering
# 
# A new feature "Age of property" is derived using Year built to deal with multimodal distribution and outliers

# In[26]:


# Create 'Age of Property'
current_year = datetime.now().year
ds['Age of Property'] = current_year - ds['Year Built']


# In[28]:


Cols_to_drop = ['Total Living Area','Year Built']
ds = ds.drop(columns=Cols_to_drop)


# In[29]:


ds.columns


# ### One- Hot Encoding for Categorical columns
# 
# In order to process categorical columns in machine learning algorithms, the categorical columns has to be encoded to numerical columns.
# 
# - Low-Cardinality Categorical Columns (≤ 20 unique values)
# These were one-hot encoded (converted into binary columns):
# 
#     - Market Region
#     - Building Type
#     - Basement
#     - Basement Finish
#     - Air Conditioning
#     - Fire Place
#     - Attached Garage
#     - Detached Garage
#     - Pool
#     - Property Use Code
# 
# - High-Cardinality Categorical Columns (> 20 unique values)
# These were frequency encoded (replaced with the count of each category's occurrence):
# 
#     - Street Type
#     - Neighbourhood Area
#     - Zoning

# In[30]:


# Drop unnamed index column if present
ds = ds.loc[:, ~ds.columns.str.contains('^Unnamed')]

# Identify categorical columns (object or bool types)
categorical_cols = ds.select_dtypes(include=['object', 'bool']).columns.tolist()

# Count unique categories in each categorical column
category_counts = ds[categorical_cols].nunique()

# Separate columns into low and high cardinality
low_cardinality = category_counts[category_counts <= 20].index.tolist()
high_cardinality = category_counts[category_counts > 20].index.tolist()

# One-hot encode low-cardinality categorical columns
df_encoded = pd.get_dummies(ds, columns=low_cardinality, drop_first=True)

# Frequency encode high-cardinality categorical columns
for col in high_cardinality:
    freq_encoding = ds[col].value_counts()
    df_encoded[col + '_freq'] = ds[col].map(freq_encoding)

# Drop original high-cardinality columns
df_encoded.drop(columns=high_cardinality, inplace=True)


# # Feature Selection, Prediction model and performance
# 
# **RF-Min-Max-ZScore and XGB-Min-Max-Zscore**
# *Actual vs Predicted* : 
# - Most predictions are clustered near the origin, indicating that the model performs well for lower-value properties.
# - As actual values increase, the predicted values tend to underperform, with many predictions falling below the ideal diagonal line.
# - This suggests the model may be underestimating high-value properties, possibly due to insufficient high valued property records in training data
# 
# 
# *Residual plot*: 
#  - Most residuals are centered around zero means predictions are mostly close to actual values. However, there’s a noticeable spread, especially for higher predicted values:
#         - Some residuals are positive, meaning the model underpredicted.
#         - Some are negative, meaning the model overpredicted.
# 
# 
# **LR-Min-Max-Zscore**
# *Actual vs Predicted* : 
# - The predictions are significantly deviating from with actual values indicating that Linear Regression is not capturing the complexity of the relationships in your data, especially with skewed or non-linear patterns.
# 
# *Residual plot*: 
# - Residuals are widely scattered, especially for higher predicted values.
# - This pattern confirms that the model is not a good fit, especially for expensive properties.
# 
# 
# **Model Performance Metrics**
# 
# R² (Coefficient of Determination): Higher is better (closer to 1).
# MAPE (Mean Absolute Percentage Error): Lower is better.
# nRMSE (Normalized Root Mean Squared Error): Lower is better.
# 
# XGBoost (XGB) performs best overall, regardless of scaling method.
# Random Forest (RF) is solid but slightly less accurate than XGB.
# Linear Regression (LR) performs poorly, with low R² and high error rates—likely underfitting the data.
# 
# 
# **Model Runtime Comparison**
# The table compares how long each model took to train and predict.
# 
# 
# Linear Regression is fastest, but its poor performance makes it unsuitable.
# XGBoost offers the best trade-off between speed and accuracy.
# Random Forest is the slowest, likely due to tree depth and ensemble size.
# 
# 
# **Model MSE Comparison**
# The table evaluates the accuracy of predictive models
# 
# XGBoost has the lowest MSE, confirming its superior predictive accuracy.
# Random Forest is close, but slightly less precise.
# Linear Regression has the highest error, reinforcing its poor fit.
# 
# 
# **Overall**
# *Best Model*: XGBoost (MinMax or ZScore)
# *Best Accuracy*: XGB (highest R², lowest MAPE & nRMSE)
# *Best Efficiency*: XGB (fast runtime, low MSE)

# In[31]:


# Drop rows with missing target values
df_encoded = df_encoded.dropna(subset=["Total Assessed Value"])

# Select numerical features only
X = df_encoded.select_dtypes(include=[np.number]).drop(columns=["Total Assessed Value"])
y = df_encoded["Total Assessed Value"]

# Define scalers and models
scalers = {
    "MinMax": MinMaxScaler(),
    "ZScore": StandardScaler()
}
models = {
    "RF": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    "XGB": XGBRegressor(random_state=42),
    "LR": LinearRegression()
}

# Store performance and runtime metrics
performance = []
runtime_results = []
mse_results = []

# Evaluate each model with each scaling method
for scale_name, scaler in scalers.items():
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()

        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        nrmse = np.sqrt(mean_squared_error(y_test, y_pred)) / (y.max() - y.min())
        runtime = round(end_time - start_time, 4)
        mse = round(mean_squared_error(y_test, y_pred), 2)

        performance.append({
            "Model": f"{model_name}-{scale_name}",
            "R²": round(r2, 3),
            "MAPE": round(mape, 3),
            "nRMSE": round(nrmse, 3)
        })
        runtime_results.append({"Model": f"{model_name}-{scale_name}", "Runtime (s)": runtime})
        mse_results.append({"Model": f"{model_name}-{scale_name}", "MSE": mse})

        # Plot Actual vs Predicted
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{model_name}-{scale_name} - Actual vs Predicted")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot Residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(6, 4))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(f"{model_name}-{scale_name} - Residuals")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Display performance results
performance_df = pd.DataFrame(performance)
runtime_df = pd.DataFrame(runtime_results)
mse_df = pd.DataFrame(mse_results)

print("Model Performance Metrics:")
display(performance_df)

print("Model Runtime Comparison:")
display(runtime_df)

print("Model MSE Comparison:")
display(mse_df)


# ### Comment:
# 
# - The table and bar chart rank the top 10 most influential features in predicting Total Assessed Value.
# - Importance scores are averaged across RF and XGB models to provide a balanced view.
# - Property type and location-related features dominate the importance rankings.
# - Water and land measurements are highly predictive, suggesting physical attributes are key drivers.

# In[32]:


# Feature importance from RF and XGB on full scaled data
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
xgb_model = XGBRegressor(random_state=42)
rf_model.fit(X_scaled, y)
xgb_model.fit(X_scaled, y)

rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_
avg_importance = (rf_importance / rf_importance.sum() + xgb_importance / xgb_importance.sum()) / 2

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "RF Importance": rf_importance / rf_importance.sum(),
    "XGB Importance": xgb_importance / xgb_importance.sum(),
    "Average Importance": avg_importance
}).sort_values(by="Average Importance", ascending=False)

# Display top 10 features
print("Top 10 Feature Importances (Average of RF and XGB):")
display(importance_df.head(10))

# Plot top 10 feature importances
plt.figure(figsize=(12, 6))
plt.bar(importance_df["Feature"][:10], importance_df["Average Importance"][:10])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Feature Importances (Average of RF and XGB)")
plt.ylabel("Normalized Importance")
plt.tight_layout()
plt.show()


# In[ ]:




