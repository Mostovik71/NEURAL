import pandas as pd
import numpy as np
from datetime import datetime
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import time
from tqdm import tqdm
from tabulate import tabulate
#BIG LENTA NEWS
'''
df1=pd.read_csv('news3NEED.csv')# news3 - 800975

#df2=pd.read_csv('news4NEED.csv')# news4 - 21673
#df2.drop(['tags'],axis=1,inplace=True)
df1.drop([0,1,2,3,4,5],inplace=True)
df1["date"]=df1["date"].apply(lambda x: (x).replace('/',','))#Переводит весь верхний регистр в нижний
df1["date"]=df1["date"].apply(lambda x:x.split(sep=','))
df1["date"]=df1["date"].apply(lambda x: list(map(int, x)))
df1["date"]=df1["date"].apply(lambda x:datetime.timestamp(datetime(x[0],x[1],x[2])))

df1.to_csv('lentaBIG.csv')
print(tabulate(df1.head(10),headers='keys'))
'''
df12=pd.DataFrame([])
df=pd.read_csv('news.csv')
df1=df[df['source']=='lenta.ru']
df2=df[df['source']=='ria.ru']
df3=df[df['source']=='meduza.io']
#print(tabulate(df1.head(10),headers='keys'))

for i,k in enumerate(tqdm(df1['title'])):#Получение строки датафрейма 1, которая похожа на какую-то строку датафрейма 2
    for j,l in enumerate((df2['title'])):

     if fuzz.partial_ratio(k, l) > 80:
      #if abs(df1['pubdate'].iloc[i] - df2['publish_date_t'].iloc[j]) < 1209600:

         df=pd.concat([pd.Series(df1['text'].iloc[i]),pd.Series(df2['text'].iloc[j])],axis=1)
         df12=df12.append(df)
         #print(df2['text'].iloc[j])
         #print(df1['text'].iloc[i])
         #print(k,l,fuzz.partial_ratio(k,l))
df12.to_csv('newsshort.csv')




