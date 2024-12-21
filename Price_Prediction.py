# -*- coding: utf-8 -*-


import pandas as pd

import numpy as np

laptop_df = pd.read_csv('laptop_data_cleaned.csv')

laptop_df.head()

laptop_df.columns

df = laptop_df.copy()

df = df.drop(columns=['Ppi'])

df.columns

df.isnull().sum()

df.dtypes

df.describe()





# Define bins and labels for weight
bins = [0, 2, 4, 6]  # Weight categories (adjust as needed)
labels = ['Light', 'Medium', 'Heavy']  # Corresponding labels

# Create a new column for weight categories
df['Weight_Category'] = pd.cut(df['Weight'], bins=bins, labels=labels)

# Display the first few rows to confirm
df[['Weight', 'Weight_Category']].head()

df

df = df.drop(columns=['Weight'])

df

categorical_columns = ['Company', 'TypeName', 'Weight_Category', 'Cpu_brand', 'Gpu_brand', 'Os']
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
df_encoded = df_encoded.astype(float)
df_encoded.head()





from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Define features and target
X = df_encoded.drop(columns=['Price'])
y = df_encoded['Price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Display evaluation metrics
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

print(df_encoded.shape)




import pickle
pickle.dump(model, open('models/model.pkl', 'wb'))
model = pickle.load(open('models/model.pkl','rb'))

