import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

el = pd.read_csv(r'C:\Users\akhil\Akhil\Coding\Linear Regression\Electric_Production (1).csv')

el['DATE'] = pd.to_datetime(el['DATE'])
el['DATE'] = el['DATE'].apply(lambda x: x.toordinal())
X = el[['DATE']]
y = el['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)


print("Accuracy:", 100 - mae)
