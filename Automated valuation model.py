#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data analysis
# 
# ## Data Loading
# 
# Installing data analysis packages/libraries and loading csv datafile using url

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

url = "https://data.winnipeg.ca/api/views/d4mq-wa44/rows.csv?accessType=DOWNLOAD"
ds = pd.read_csv(url)
ds


# In[2]:


# displaying summary of the data
ds.info()


# 

# In[3]:


# Identify columns with 0 non-null values
empty_columns = ds.columns[ds.isnull().all()].tolist()

# Drop all empty columns except the one to keep
columns_to_drop = [col for col in empty_columns ] #if col != column_to_keep
ds = ds.drop(columns=columns_to_drop)


# ### Comments
# 
# Identified few numerical columns that were of data type 'object'. 
# They have some speical characters like Dollar sign, comma etc that needs to be stripped.
# 
# After scrubbing, the descriptive statistics of numerical columns is as following,

# ## Descriptive statistics for numerical columns

# In[5]:


# Numeric columns with special characters that need cleaning
columns_to_clean = ['Street Number','Total Living Area','Total Assessed Value','Assessed Land Area','Water Frontage Measurement','Sewer Frontage Measurement',
                    'GISID','Roll Number','Dwelling Units','Current Assessment Year']

# Add median separately since it's not included in describe()
descriptive_stats.loc['median'] = ds.median()

# Remove the dollar sign and convert to float
ds[columns_to_clean] = ds[columns_to_clean].replace('[\$,]', '', regex=True).astype(float)
ds.describe()


# # Comments:
# 
# Identified two outliers row in the file while getting descriptive statistics of the columns

# In[6]:


# Changing the data type of 'Roll Number' to object in order to remove the long random roll number
ds['Roll Number'] = ds['Roll Number'].astype(object)

# List of record IDs to drop
roll_nos_to_drop = ['0107376627423 49.80160863413996', '9257376']

# Drop rows where record_id is in the list
ds = ds[~ds['Roll Number'].isin(roll_nos_to_drop)]


# ## Descriptive statistics of categorical columns

# In[9]:


# Generate descriptive statistics for categorical columns
categorical_stats = ds.describe(include='object')
categorical_stats


# ## Comments:
# 
# While going through the dataset, two rows were identified with arbitary values, '0107376627423 49.80160863413996' in all the columns, so they were dropped.

# # Handling missing data
# 
# Numerical variables like Total Living Area, Rooms, Water Frontage Measurement, Sewer Frontage Measurement, Total Assessed Value were replaced by Median
# 
# Categorical Variable like Street Type, Building type, Basement, Basement Finish, Year Built were replaced by mode or by introducing new category "Missing". 

# In[10]:


# calculating number of null values in each column
ds.isnull().sum()


# In[11]:


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
# The assignement asked for AVM for RESIDENTIAL PROPERTIES. So the column 'Property Use Code' is used to identify residential properties. 

# In[12]:


# List all unique categories in the 'Property Use Code' column to identify residential properties only
unique_categories = ds['Property Use Code'].dropna().unique().tolist()
unique_categories


# In[13]:


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

# In[14]:


# Select only numerical features for correlation analysis
numerical_dataset = ds.select_dtypes(include=['number'])
correlation=numerical_dataset.corr()
print(correlation['Total Assessed Value'].sort_values(ascending=False),'\n')


# # Comment:
# 
# As per correlation analysis, dropping column 'Current Assessment Year', 'GISD','Street number', 'Centroid Lon', 'Centroid Lat'.
# 

# In[15]:


Cols_to_drop = ['Street Number','Current Assessment Year','GISID','Centroid Lon','Centroid Lat']
ds = ds.drop(columns=Cols_to_drop)


# In[16]:


# Select only numerical features for correlation analysis
numerical_dataset = ds.select_dtypes(include=['number'])

plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)


# # Comments:
# 
# Identifying object, integer, float type columns and assigning them to different subsets

# In[17]:


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


# # Comments: 
# identifying number of unique categores each object type columns have.

# In[18]:


unique_values = []
for col in object_cols:
    unique_values.append(ds[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)


# In[19]:


category_counts = ds[object_cols].nunique()
category_counts


# # Comments:
#  - Features with a large number of unique categories can be challenging to handle in machine learning models, as they might lead to high-dimensional data.The columns with more than 500 categories may not contribute to AVM. Hence they can be dropped. 
#  - Also columns with 1 category were looked into manually in excel file. They were just sigular value against a proprty which would play no role in prediction modelling, so they are dropped.
#  - Columns like Assessed Value, status, Property Class 1, 2,3,4,5 were dropped as we have a condensed column displaying the same values.

# In[20]:


Cols_to_drop = ['Full Address','Geometry','Detail URL',
                'Assessed Value 1','Assessed Value 2','Assessed Value 3','Assessed Value 4',
                'Assessed Value 5','Property Class 1','Property Class 2','Property Class 3',
                'Property Class 4','Property Class 5','Status 1','Status 2','Status 3','Status 4',
                'Status 5','Roll Number','Unit Number','Street Name','Street Suffix','Dwelling Units',
                'Multiple Residences','Property Influences','Number Floors (Condo)']
ds = ds.drop(columns=Cols_to_drop)


# In[28]:


ds


# # One- Hot Encoding for Categorical columns
# 
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

# In[21]:


from sklearn.preprocessing import LabelEncoder

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

# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt



# Drop rows with missing target values
df_encoded = df_encoded.dropna(subset=["Total Assessed Value"])

# Select numerical features only
numerical_features = df_encoded.select_dtypes(include=[np.number]).drop(columns=["Total Assessed Value"])
X = numerical_features
y = df_encoded["Total Assessed Value"]

# Define scalers
scalers = {
    "MinMax": MinMaxScaler(),
    "ZScore": StandardScaler()
}

# Define models
models = {
    "RF": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    "XGB": XGBRegressor(random_state=42),
    "LR": LinearRegression()
}

# Store performance metrics
performance = []

# Evaluate each model with each scaling method
for scale_name, scaler in scalers.items():
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        nrmse = np.sqrt(mean_squared_error(y_test, y_pred)) / (y.max() - y.min())

        performance.append({
            "Model": f"{model_name}-{scale_name}",
            "R²": round(r2, 3),
            "MAPE": round(mape, 3),
            "nRMSE": round(nrmse, 3)
        })

# Create performance comparison table
performance_df = pd.DataFrame(performance)

# Save the table to CSV
performance_df.to_csv("model_performance_comparison.csv", index=False)

# Plot feature importances for RF and XGB
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
xgb_model = XGBRegressor(random_state=42)

rf_model.fit(X_scaled, y)
xgb_model.fit(X_scaled, y)

rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_

# Normalize importances
rf_norm = rf_importance / rf_importance.sum()
xgb_norm = xgb_importance / xgb_importance.sum()

# Average importance
avg_importance = (rf_norm + xgb_norm) / 2

# Create DataFrame for plotting
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "RF Importance": rf_norm,
    "XGB Importance": xgb_norm,
    "Average Importance": avg_importance
}).sort_values(by="Average Importance", ascending=False)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(importance_df["Feature"][:10], importance_df["Average Importance"][:10])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Feature Importances (Average of RF and XGB)")
plt.ylabel("Normalized Importance")
plt.tight_layout()
plt.savefig("feature_importance_comparison.png")

# Save importance table
importance_df.to_csv("feature_importance_comparison.csv", index=False)


# In[23]:


import time

# Store runtime and performance metrics
runtime_results = []
mse_results = []

# Evaluate each model
for model_name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()

    # Calculate runtime and MSE
    runtime = round(end_time - start_time, 4)
    mse = round(mean_squared_error(y_test, y_pred), 2)
    runtime_results.append({"Model": model_name, "Runtime (s)": runtime})
    mse_results.append({"Model": model_name, "MSE": mse})

    # Scatter plot: Actual vs Predicted
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_actual_vs_predicted.png")

    # Scatter plot: Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"{model_name} - Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_residuals.png")

# Save runtime and MSE results to CSV
pd.DataFrame(runtime_results).to_csv("model_runtime_comparison.csv", index=False)
pd.DataFrame(mse_results).to_csv("model_mse_comparison.csv", index=False)


# In[ ]:




