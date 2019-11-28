# --------------
# Import Libraries
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)

print(df.head())
# Map the lowering function to all column names
df.columns = map(str.lower, df.columns)
df.columns = df.columns.str.replace(" ","_")   #Remove blank Space
df.replace('NaN', np.nan) 
# Code ends here


# --------------
from sklearn.model_selection import train_test_split
df.set_index(keys='serial_number',inplace=True,drop=True)


# Code starts
df[['established_date', 'acquired_date']] = df[['established_date', 'acquired_date']].apply(pd.to_datetime)
X = df.iloc[:,:-1]
y = df['2016_deposits']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state = 3)
# Code ends here


# --------------
# time_col = X_train.select_dtypes(exclude=[np.number,'O']).columns
from datetime import date 
time_col = ['acquired_date','established_date']

# Code starts here
for col_name in time_col:
    new_col_name = 'since_'+ col_name

X_train[new_col_name] = (pd.datetime.now() - X_train[col_name]) 

X_train[new_col_name] = X_train[new_col_name].apply(lambda x: float(x.days)/365)
X_train.drop(columns=['established_date'], inplace = True)
print(X_train['since_established_date'][100])
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
cat = X_train.select_dtypes(include='O').columns.tolist()

# Code starts here
X_train.replace(np.nan, 0,  inplace= True)
X_val.replace(np.nan, 0, inplace= True)
le = LabelEncoder()
# X_train[cat] = le.fit_transform(X_train[cat])
# X_val[cat] = le.fit_transform(X_val[cat])

X_train_temp = pd.get_dummies(X_train)
X_val_temp = pd.get_dummies(X_val)
# Code ends here


# --------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
# Code starts here

# create a regressor object 
dt = DecisionTreeRegressor(random_state = 5)  
  
# fit the regressor with X and Y data 
dt.fit(X_train, y_train) 
y_pred = dt.predict(X_val)
accuracy = dt.score(X_val, y_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))


# --------------
from xgboost import XGBRegressor


# Code starts here

xgb = XGBRegressor(max_depth=50, learning_rate=0.83, n_estimators=100)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_val)
accuracy = xgb.score(X_val, y_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
# Code ends here


