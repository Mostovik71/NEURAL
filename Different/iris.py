import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
iris = datasets.load_iris()
iris_frame = pd.DataFrame(iris.data)
iris_frame.columns = iris.feature_names
iris_frame['target'] = iris.target
iris_frame['name'] = iris_frame.target.apply(lambda x : iris.target_names[x])

iris_frame.drop(['name'],axis=1,inplace=True)


X = iris_frame.drop(['target'], axis=1)
y = iris_frame['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,  test_size=0.33, random_state=42)
print(X_train)
plt.scatter(X_train['sepal width (cm)'],X_train['petal width (cm)'])
plt.show()

#print(lr.coef_)#Коэффициенты логистической регрессии
#print(lr.classes_)#Список классов, на которые делит логистическая регрессия
#print(lr.class_weight)
#df=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_pred)],axis=1,ignore_index=True)

#Логистическая регрессия
'''
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred_proba=lr.predict_proba(X_test)
df1=pd.DataFrame(y_test)
df2=pd.DataFrame(y_pred)
df3=pd.DataFrame(y_pred_proba)
#print(df3)
df1.reset_index(drop=True, inplace=True)
df=pd.concat([df1,df2,df3],axis=1)
print(df)
#print(tabulate(pd.DataFrame(y_test),headers='keys'))
'''

#Линейная регрессия
'''
linear=LinearRegression()
linear.fit(X_train,y_train)
y_pred_linear=linear.predict(X_test)
print(linear.coef_,lr.coef_)
regr=[round(i) for i in y_pred_linear.tolist()]
df1=pd.DataFrame(y_test)
df2=pd.DataFrame(regr)
df1.reset_index(drop=True, inplace=True)
df3=pd.concat([df1,df2],axis=1)
print(tabulate(df3,headers='keys'))
'''
#ROC-AUC кривая
'''
sns.set(font_scale=1.5)
sns.set_color_codes("muted")

plt.figure(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1], pos_label=1)
lw = 2
plt.plot(fpr, tpr, lw=lw, label='ROC curve ')
plt.plot([0, 1], [0, 1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.savefig("ROC.png")
'''


#Random Forest
rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(X, y)
y_pred_rf=rf.predict(X_test)
df1=pd.DataFrame(y_test)
df2=pd.DataFrame(y_pred_rf)
df1.reset_index(drop=True, inplace=True)
df=pd.concat([df1,df2],axis=1)
print(df)
print(rf.score(X_test,y_pred_rf))