import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

el = pd.read_csv(r'/home/astrix/Code/Elec/Electric_Production.csv')

el['DATE'] = pd.to_datetime(el['DATE']).apply(lambda x: x.toordinal())
el["Value"].replace(0,np.NaN, inplace=True)
el['Value'].fillna(el['Value'].mean(), inplace=True)

X = el[['DATE']]
y = el['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

el['Prediction'] = model.predict(X)
el['DATE'] = el['DATE'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))
print(el)
print(100-mae)

plt.plot(X)

plt.plot(el['DATE'], el['Value'], label='Actual')
plt.plot(el['DATE'], el['Prediction'], label='Prediction')
plt.xlabel('Date')
plt.ylabel('Electricity Production')
plt.title('Actual vs. Predicted Electricity Production')
plt.legend()
plt.show()
