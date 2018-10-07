import matplotlib.pyplot as py 
import pandas as pd
import seaborn as sb

df=sb.load_dataset('tips')
df.head()
df.info()
df.describe()
df.sample(5)
df.groupby('day').count()

df2=df.groupby('day').sum() # sum per day
df2.drop('size',inplace=True,axis=1) # sum of size column is not relevant
df2['percent'] = df2['tip']/df2['total_bill']*100 # add percents

df3=df.groupby('smoker').sum()
df3['percent'] = df3['tip']/df3['total_bill']*100

df4= df.groupby(['day','size']).sum()
df4['percent'] = df4['tip']/df4['total_bill']*100
df4.dropna() # drop null rows


sb.countplot(x='day', data=df)

sb.countplot(x='day', hue='size',data=df)

sb.countplot(x='day', hue='smoker', data=df)

#convert sex and smoker columns to values
df.replace({ 'sex': {'Male':0 , 'Female':1} , 'smoker' : {'No': 0 , 'Yes': 1}} ,inplace=True)
df.head()

#Using dummy variables
days=pd.get_dummies(df['day'])
days.sample(5)

days=pd.get_dummies(df['day'],drop_first=True)
days.sample(6)

days=pd.get_dummies(df['day'],drop_first=True)
df = pd.concat([df,days],axis=1)
times=pd.get_dummies(df['time'],drop_first=True)
df = pd.concat([df,times],axis=1)
df.drop(['day','time'],inplace=True,axis=1)
df.head()

X = df[['sex','smoker','size','Fri','Sat','Sun','Dinner']]
Y = df[['tip']]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test , y_train , y_test = train_test_split(X,Y,test_size=0.25,random_state=26)

model = LinearRegression()
model.fit(X_train, y_train)

predictions= model.predict(X_test)

sb.distplot(y_test-predictions)