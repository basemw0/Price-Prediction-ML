import pandas as pd
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , PolynomialFeatures
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , StandardScaler
from sklearn import linear_model as lm
from xgboost import XGBRegressor
data = pd.read_csv("train.csv" )
data = data.drop(['X1','X10','X3', 'X5','X8'],axis=1)
data['X2'] = data['X2'].fillna(data['X2'].mean())
data['X9'] = data['X9'].fillna(data["X9"].mode()[0])

le = LabelEncoder()

fitted = le.fit(data["X9"])
data["X9"] = fitted.transform(data["X9"])

#One hot X11
oneHotEncoder = OneHotEncoder()
X11OneHotFit = oneHotEncoder.fit(data.X11.values.reshape(-1, 1))
X11OneHot =  oneHotEncoder.transform(data.X11.values.reshape(-1, 1)).toarray()

dfOneHotX11 = pd.DataFrame(X11OneHot , columns= ["X11_"+str(int(i)) for i in range(X11OneHot.shape[1])])
data = pd.concat([data,dfOneHotX11],axis=1)

#Drop x112 x113 dont correlate
data = data.drop(['X11_2',"X11_3"] , axis=1)
#One hot X7
oneHotEncoder7 = OneHotEncoder()
X7OneHotFit = oneHotEncoder7.fit(data.X7.values.reshape(-1, 1))
X7OneHot =  oneHotEncoder7.transform(data.X7.values.reshape(-1, 1)).toarray()
dfOneHotX7 = pd.DataFrame(X7OneHot , columns= ["X7_"+str(int(i)) for i in range(X7OneHot.shape[1])])
data = pd.concat([data,dfOneHotX7],axis=1)
data = data.drop(["X7","X11"],axis=1)

#drop X11 to view X7
data = data.drop(["X7_4","X7_5","X7_6","X7_7","X7_8","X7_9","X7_1","X7_2","X7_3"] , axis=1)

XToNormal = data[["X2","X6"]]
norm  = MinMaxScaler().fit(XToNormal)
Xnormalized = norm.transform(XToNormal)
X_Dataframe_Norm = pd.DataFrame(Xnormalized, columns=XToNormal.columns)
for column in X_Dataframe_Norm : 
    data[column] = X_Dataframe_Norm[column]
# print(scaleX4)
Xtrain=data.drop("Y",axis=1)
# print(Xtrain.head())
Ynum = np.array(data["Y"])
# print(data.corr())

# Load test data
data2 = pd.read_csv("test.csv")

# Drop unnecessary columns
data2 = data2.drop(['X1', 'X10', 'X3', 'X5', 'X8'], axis=1)
data2['X2'] = data2['X2'].fillna(data2['X2'].mean())
data2['X9'] = data2['X9'].fillna(data2['X9'].mode()[0])

# Label encode X9

data2['X9'] = fitted.transform(data2['X9'])
# One-hot encode X11
X11OneHot = oneHotEncoder.transform(data2.X11.values.reshape(-1, 1)).toarray()
dfOneHotX11 = pd.DataFrame(X11OneHot, columns=["X11_" + str(int(i)) for i in range(X11OneHot.shape[1])])
data2 = pd.concat([data2, dfOneHotX11], axis=1)

# Drop unnecessary X11 columns and X11 itself
data2 = data2.drop(['X11_2', 'X11_3', 'X11'], axis=1)

# One-hot encode X7
X7OneHot = oneHotEncoder7.transform(data2.X7.values.reshape(-1, 1)).toarray()
dfOneHotX7 = pd.DataFrame(X7OneHot, columns=["X7_" + str(int(i)) for i in range(X7OneHot.shape[1])])
data2 = pd.concat([data2, dfOneHotX7], axis=1)

# Drop unnecessary X7 columns and X7 itself
data2 = data2.drop(['X7', "X7_4", "X7_5", "X7_6", "X7_7", "X7_8", "X7_9", "X7_1", "X7_2", "X7_3"], axis=1)

# Normalize X2 and X6
XToNormal = data2[["X2", "X6"]]
Xnormalized = norm.transform(XToNormal)
X_Dataframe_Norm = pd.DataFrame(Xnormalized, columns=XToNormal.columns)

for column in X_Dataframe_Norm:
    data2[column] = X_Dataframe_Norm[column]
print(data.head())
# Prepare final test dataset
# print(data2.head())
# print(data2.head())
model = XGBRegressor(n_estimators=200, learning_rate=0.3, max_depth=6, random_state=42)
model.fit(Xtrain, Ynum)
prediction = model.predict(np.array(data2))
submitions = pd.DataFrame({
        'row_id' :data2.index,
        'Y' : prediction
})
print(prediction)
submitions.to_csv('sample_submission.csv', index=False)




