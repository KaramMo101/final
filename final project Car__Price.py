#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, roc_auc_score
import joblib
import streamlit as st
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# In[2]:


#Dataset link:
#https://www.kaggle.com/datasets/abdo977/used-car-price-in-egypt


# Read the file
df = pd.read_csv(r"C:\Users\RTX\OneDrive\Desktop\Final project\Cars.csv")

df.head()


# In[3]:


df.info()


# In[4]:


#unique_brands = df["Brand"].unique()
#print(unique_brands)


# In[5]:


# Check for duplicate rows
duplicated_rows = df.duplicated()
print('Number of duplicate rows = ', duplicated_rows.sum())

# Check for missing values
missing_values = df.isnull().sum()
print('Number of missing values in each column = ', missing_values)


# In[6]:


# Drop the unnamed column and duplicates
df = df.drop(columns=['Unnamed: 0'])
df.head()


# In[7]:


# Display statistical analysis for numerical data
df.describe()


# In[8]:


# Perform univariate and bivariate analysis on numeric columns only
df.select_dtypes(include=[np.number]).describe()
df.select_dtypes(include=[np.number]).corr()


# In[9]:


# Display statistical analysis for non-numerical data
df.describe(include=['O'])


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plotting the distribution of numerical columns
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Plotting boxplots for numerical columns to detect outliers
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# In[11]:


from scipy.stats import zscore

# Calculate z-scores of `df`
z_scores = zscore(df.select_dtypes(include=['int64', 'float64']))

# Define a threshold to identify an outlier
threshold = 3

# Get boolean mask where z-score greater than the threshold
mask = (np.abs(z_scores) < threshold).all(axis=1)

# Remove outliers
df = df[mask]

# Normalize the numerical columns
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = (df[col] - df[col].mean()) / df[col].std()


# In[12]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Read the file
df = pd.read_csv(r"C:\Users\RTX\OneDrive\Desktop\Final project\Cars.csv")

# Drop the unnamed column and duplicates
df.drop(columns=['Unnamed: 0'], inplace=True)
df.drop_duplicates(inplace=True)

# Display statistical analysis for numerical data
numerical_stats = df.describe()

# Display statistical analysis for non-numerical data
categorical_stats = df.describe(include=['O'])

# Plotting the distribution of numerical columns
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Plotting boxplots for numerical columns to detect outliers
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Calculate z-scores of numerical columns
z_scores = zscore(df.select_dtypes(include=['int64', 'float64']))

# Define a threshold to identify an outlier
threshold = 3

# Get boolean mask where z-score greater than the threshold
mask = (np.abs(z_scores) < threshold).all(axis=1)

# Remove outliers
df = df[mask]

# Normalize the numerical columns
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = (df[col] - df[col].mean()) / df[col].std()


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot for numerical columns
df_numerical = df.select_dtypes(include=['int64', 'float64'])
sns.pairplot(df_numerical)
plt.show()

# Countplot for categorical columns
df_categorical = df.select_dtypes(include=['object'])
for col in df_categorical.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=df_categorical)
    plt.title(f'Countplot of {col}')
    plt.show()

# Correlation heatmap for numerical columns
corr = df_numerical.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# ## The question is "What is the predicted price of a car given its features?"

# In[14]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Define categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Define numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

# Define models
models = {
    'lr': LinearRegression(),
    'dt': DecisionTreeRegressor(random_state=0),
    'svr': SVR(),
    'xgb': XGBRegressor(random_state=0),
    'cat': CatBoostRegressor(verbose=0, random_state=0)
}


# In[15]:


from sklearn.model_selection import train_test_split

# Define features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Split data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[16]:


# Print the column names in categorical_cols and numerical_cols
#print("Categorical columns:", categorical_cols)
#print("Numerical columns:", numerical_cols)

# Print the column names in X_train
#print("Columns in X_train:", X_train.columns)


# In[ ]:





# In[17]:


# Remove outliers
df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]


# In[18]:


# Feature Engineering
X = df.drop(columns=['Price'])
y = df['Price']


# In[19]:


# Preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])


# In[20]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


# Define preprocessor
numeric_features = df.select_dtypes(include=['int64', 'float64']).drop(['Price'], axis=1).columns
categorical_features = df.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# Define the pipelines
pipelines = [
    ('Linear Regression', Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])),
    ('Decision Tree', Pipeline([('preprocessor', preprocessor), ('model', DecisionTreeRegressor())])),
    ('SVR', Pipeline([('preprocessor', preprocessor), ('model', SVR())])),
    ('Gradient Boosting', Pipeline([('preprocessor', preprocessor), ('model', GradientBoostingRegressor())])),
    ('XGBoost', Pipeline([('preprocessor', preprocessor), ('model', XGBRegressor())])),
    ('CatBoost', Pipeline([('preprocessor', preprocessor), ('model', CatBoostRegressor(silent=True))]))
]



# In[22]:


# Split your data into training and testing data
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the models and print the results
for name, pipeline in pipelines:
    pipeline.fit(X_train, y_train)
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(pipeline, X, y, cv=kfold).mean()
    print(f'Model: {name}')
    print(f'Training Score: {train_score}')
    print(f'Testing Score: {test_score}')
    print(f'Cross-Validation Score: {cv_score}\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Select the best model
best_model = max(pipelines, key=lambda item:item[1].score(X_test, y_test))
print(f'Best model: {best_model[0]}')



# In[24]:


get_ipython().run_cell_magic('writefile', 'Car_Price.py', '\nimport streamlit as st\nimport joblib\nimport pandas as pd\n\n# Load the model and the inputs\nmodel = joblib.load(\'best_model.pkl\')\ndf1 = pd.read_csv(r"C:\\Users\\RTX\\OneDrive\\Desktop\\Final project\\Cars.csv")\n\ndef user_input_features(df):\n    """Collect user inputs for car features"""\n    # Define the features\n    Brand = st.selectbox(\'Brand\', options=df[\'Brand\'].unique())\n    Model = st.selectbox(\'Model\', options=df[\'Model\'].unique())\n    Body = st.selectbox(\'Body\', options=df[\'Body\'].unique())\n    Color = st.selectbox(\'Color\', options=df[\'Color\'].unique())\n    Year = st.slider(\'Year\', min_value=int(df[\'Year\'].min()), max_value=int(df[\'Year\'].max()), value=int(df[\'Year\'].mean()))\n    Fuel = st.selectbox(\'Fuel\', options=df[\'Fuel\'].unique())\n    Kilometers = st.selectbox(\'Kilometers\', options=df[\'Kilometers\'].unique())\n    Engine = st.selectbox(\'Engine\', options=df[\'Engine\'].unique())\n    Transmission = st.selectbox(\'Transmission\', options=df[\'Transmission\'].unique())\n    Gov = st.selectbox(\'Gov\', options=df[\'Gov\'].unique())\n\n    # Create a data frame from the inputs\n    data = {\n        \'Brand\': [Brand],\n        \'Model\': [Model],\n        \'Body\': [Body],\n        \'Color\': [Color],\n        \'Year\': [Year],\n        \'Fuel\': [Fuel],\n        \'Kilometers\': [Kilometers],\n        \'Engine\': [Engine],\n        \'Transmission\': [Transmission],\n        \'Gov\': [Gov]\n    }\n    features = pd.DataFrame(data)\n    return features\n\ndef predict(model, input_df):\n    """Make predictions and display them"""\n    prediction = model.predict(input_df)\n    st.write(f\'The predicted price for this {input_df["Brand"][0]} {input_df["Model"][0]} car is \', prediction[0])\n\ndef main():\n    """Main function to run the app"""\n    st.title(\'Car Price Prediction App\')\n    st.write(\'Enter the details of the car and get the predicted price!\')\n    input_df = user_input_features(df1)\n    predict(model, input_df)\n\nif __name__ == \'__main__\':\n    main()\n')


# In[25]:


#streamlit run Car_Price.py


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




