# Automated-Valuation-Model
## Technical Description

A simple automation model is build to predict the assessed value of a residential property in Winnipeg. The following steps were followed to build this automation:

**1. Data Collection**
The csv data was extracted from [Assessment dataset] https://data.winnipeg.ca/Assessment-Taxation-Corporate/Assessment-Parcels/d4mq-wa44/about_data. 

**2. Data Pre-Processing**
•	Drops empty and irrelevant columns.
•	Cleans numeric columns with special characters (e.g., $, ,).
•	Handles missing values using median (numerical) and mode or "Missing" (categorical).
•	Filters for residential properties using Property Use Code.


### Assumtions:
1. The dataset extracted from the above mentioned website was a large file. I used a random subset of the file to tackle the problem.
2. 


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/prarnamehta/AVM/HEAD?filepath=Automated%20valuation%20model.ipynb)
https://automated-valuation-model.streamlit.app/
