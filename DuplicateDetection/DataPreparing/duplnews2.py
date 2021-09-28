import pandas as pd
import numpy as np
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from tqdm import tqdm
from tabulate import tabulate
df1=pd.read_csv('news1NEED.csv')# news1 и news2 - rbc и lenta
df2=pd.read_csv('news2NEED.csv')
df12=pd.DataFrame([])
#df3=pd.read_csv('news3NEED.csv')
#df4=pd.read_csv('news4NEED.csv')
df1.dropna(inplace=True)
df2.dropna(inplace=True)
df1=df1.sort_values(by='pubdate')
df2=df2.sort_values(by='publish_date_t')
#df1.drop(['pubdate'],axis=1,inplace=True)
df1=df1[df1['pubdate']>970130900]
#df2=df2[df2['publish_date_t']>970144467]

#df2.drop(['publish_date_t','publish_date'],axis=1,inplace=True)
#print(tabulate(df1.head(100),headers='keys'))
#print(tabulate(df2.head(100),headers='keys'))
#df1=df1.head(10000)



#Есть 4 датасета с новостями. С помощью fuzzywuzzy проверить по заголовкам на совпадения, и обьединить все это в один датасет
for i,k in enumerate(tqdm(df2['title'])):#Получение строки датафрейма 1, которая похожа на какую-то строку датафрейма 2
    for j,l in enumerate((df1['title'])):

     if abs(df2['publish_date_t'].iloc[i]-df1['pubdate'].iloc[j]) < 86400:
      if fuzz.partial_ratio(k, l) > 80:


         df=pd.concat([pd.Series(df1['text'].iloc[i]),pd.Series(df2['text'].iloc[j])],axis=1)
         df12=df12.append(df)

         #print(df2['text'].iloc[j])
         #print(df1['text'].iloc[i])
         #print(k,l,fuzz.partial_ratio(k,l))


df12.to_csv('newstest.csv')


