import pandas as pd
import numpy as np
from sklearn import linear_model
import plotly.express as px
import pingouin as pg

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

corrFrame = df_train.corr(method='pearson')
corrMatrix = px.imshow(corrFrame, text_auto=True)
corrMatrix.show()  # squareMeters and numberOfRooms have the highest correlation

print(np.where(df_train['squareMeters'] > 100000))
df_train = df_train.drop(4741)
df_train = df_train.drop(15334)
print(df_train.describe())

scatterSquareMeters = px.scatter(x=df_train['squareMeters'], y=df_train['price'], trendline='ols')
scatterSquareMeters.show()

results = px.get_trendline_results(scatterSquareMeters)
results = results.iloc[0]['px_fit_results'].summary()
print(results)
###################################################
# Creating a linear regression model with just squareMeters

regression = linear_model.LinearRegression()

X_train = df_train[['squareMeters']]
y_train = df_train[['price']]
X_test = df_test[['squareMeters']]

model = regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)
y_pred = np.around(y_pred, decimals=1)

df_sub = df_test[['id']].copy()
df_sub['price'] = y_pred

df_sub.to_csv('submission1_Feb17.csv', index=False, sep=',')

# scatterNumOfRooms = px.scatter(x=df_train['numberOfRooms'], y=df_train['price'], trendline='ols')
# scatterNumOfRooms.show()

# hasYardBox = px.box(df_train, x='hasYard', y='price')
# hasYardBox.show()  # Does not appear to be significant

# hasPoolBox = px.box(df_train, x='hasPool', y='price')
# hasPoolBox.show() # hasPool does not seem to be significant
#
# poolGroup = df_train[df_train['hasPool'] == 1]
# noPoolGroup = df_train[df_train['hasPool'] == 0]
# print(pg.ttest(poolGroup['price'], noPoolGroup['price']))
