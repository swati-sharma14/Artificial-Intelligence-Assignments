from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import statsmodels.api as sm
from patsy.contrasts import Helmert
from sklearn.preprocessing import StandardScaler

  
# fetch dataset 
abalone = fetch_ucirepo(id=1) 
  
# data (as pandas dataframes) 
X = abalone.data.features 
y = abalone.data.targets 

def func(x):
   return 1/(1 + (np.exp(x)))

# Separate the continuous variables for scaling
continuous_vars = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
X_continuous = X[continuous_vars]

# Apply standard scaling to the continuous variables
scaler = StandardScaler()
X_continuous_scaled = scaler.fit_transform(X_continuous)

# Combine the scaled continuous variables with the categorical variable 'Sex'
X_scaled = pd.concat([pd.get_dummies(X['Sex'], prefix='Sex', drop_first=True), pd.DataFrame(X_continuous_scaled, columns=continuous_vars)], axis=1)
X_encoded = X_scaled.copy()

for column in X_encoded:
    X_encoded[column] = X_encoded[column].apply(func)
    
X_encoded['Length'] = np.exp(X_encoded['Length'])
X_encoded['Diameter'] = np.square(X_encoded['Diameter'])
X_encoded['Height'] = np.square(X_encoded['Height'])

for column in X_encoded:
    X_encoded[column] = X_encoded[column].apply(func)

# Split the encoded data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2)

# Train a linear regression model
# Evaluate the model's performance using the entire dataset
model = LinearRegression()
model.fit(X_encoded, y)
r2 = r2_score(y, model.predict(X_encoded))
print("Full dataset train and eval R2 score:", r2)

# Evaluate the model's performance using cross-validation
n_splits = 100
r2_scores = []

for _ in range(n_splits):
    X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.15)
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2_score_val = r2_score(y_val, model.predict(X_val))
    r2_scores.append(r2_score_val)

print("70-15-15 Cross validation boxplot: mean={:.15f}, std={:.15f}".format(sum(r2_scores) / len(r2_scores), np.std(r2_scores)))