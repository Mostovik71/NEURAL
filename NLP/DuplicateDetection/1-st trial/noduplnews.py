import pandas as pd
from tqdm import tqdm
import re
from fuzzywuzzy import fuzz
from tabulate import tabulate
#df1=pd.read_csv('news1NEED.csv',nrows=1500)# news1 и news2 - rbc и lenta
df1=pd.read_csv('news.csv').sample(500,random_state=42)
df12=pd.DataFrame([])
#df1.dropna(inplace=True)
#df2.dropna(inplace=True)
for i,k in enumerate(tqdm(df1['title'])):#Получение строки датафрейма 1, которая похожа на какую-то строку датафрейма 2
    for j,l in enumerate((df1['title'])):



      if fuzz.partial_ratio(k, l) <50:


         df=pd.concat([pd.Series(df1['title'].iloc[j]),pd.Series(df1['text'].iloc[i])],axis=1)
         df12=df12.append(df)
         i+=1
         j+=1
         #print(df2['text'].iloc[j])
         #print(df1['text'].iloc[i])
         #print(k,l,fuzz.partial_ratio(k,l))


df12.to_excel('summ.xlsx')