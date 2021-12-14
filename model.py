import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('ubersuggest_google.csv')
dt=df.copy()

x=dt.iloc[:,1:3]
y=dt.iloc[:,3:4]
x1=dt.iloc[:,4:]
x=pd.concat([x,x1], axis=1, join='inner')

labelencoder = LabelEncoder()
x['Keyword'] = labelencoder.fit_transform(x['Keyword'])

y=y.CPC.str.replace("[$]", "", regex=True)
y=y.astype(float)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)

# fiting the model
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
yhat = dt.predict(x_test)

# predicting the value
s=labelencoder.transform(['google timer'])
print(s[0])
sample=[s[0],74000,1,41]
pred=dt.predict([sample])
print(pred)

# creating the pickle file for labelencoding
pickle.dump(labelencoder,open('labelencoder_model.pkl','wb'))


# creating the pickle file for DecisionTreeRegressor
pickle.dump(dt,open('dt_model.pkl','wb'))