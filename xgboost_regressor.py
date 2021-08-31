
"""XGBoost Regressor

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

"""## Importing the dataset"""

dataset = pd.read_csv('Binance Coin - Historic data.csv')

dataset.head()
print(dataset.columns)
dataset.info()

for x in range(dataset['Vol.'].shape[0]):
    val = dataset["Vol."].iloc[x]
    if "M" in val:
        dataset["Vol."].iloc[x] = float(val[:-1]) * 1000000
        pass
    elif "K" in val:
        dataset["Vol."].iloc[x] = float(val[:-1]) * 1000
dataset["Vol."]

dataset['Vol.']=pd.to_numeric(dataset['Vol.'],errors='coerce')
dataset.info()

dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['month'] = dataset['Date'].dt.month
dataset['year'] = dataset['Date'].dt.year
dataset['month_year'] = dataset['Date'].dt.to_period('M')
dataset

import seaborn as sns
sns.set(rc = {'figure.figsize' : (14,12)})
sns.heatmap(dataset.corr().round(2) , annot = True )

dataset = dataset.drop(columns=['Vol.','Change%','month'])
dataset

X = np.array(dataset.iloc[ : , 2 : 5].values)
y = dataset.iloc[:, 1].values

print(X)

"""##Splitting data into Training and Testing"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""##*Training the XGBoost model on the Training set*"""

from xgboost import XGBRegressor
regressor = XGBRegressor(objective ="reg:squarederror")
regressor.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))

score = regressor.score(X_train, y_train)  
print("Training score: ", score)
scores = cross_val_score(regressor, X_train, y_train,cv=10)
print("Mean cross-validation score: %.2f" % scores.mean())

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

plt.figure(figsize = (16,8))
 plt.plot(y_pred, label='Predicted Values')
 plt.plot(y_test, label='True/Correct values')
 plt.legend(loc='best')
 plt.show()
