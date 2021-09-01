import pandas as pd
from tabulate import tabulate
import pickle
import re
import string
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

df=pd.read_excel('forlog1024.xlsx')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.dropna(inplace=True)





y_train = df['is_duplicate']
df.drop('is_duplicate', axis=1, inplace=True)
x_train = df

lr = LogisticRegression(max_iter=1700)
lr.fit(x_train, y_train)
pkl_filename = "logreg1024.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(lr, file)
#with open('logreg.pkl', 'rb') as file:
#    lr = pickle.load(file)
#df['duplicate']=pd.Series(lr.predict_proba(df)[:,1])
#print(tabulate(df.head(110),headers='keys'))







