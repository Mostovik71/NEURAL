import pandas as pd
import re
import string
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
df=pd.read_excel('some xl.xlsx')
df1=df[['row']]
df1["row"]=df1["row"].apply(lambda x: x.lower())#Переводит весь верхний регистр в нижний
df1["row"]=df1["row"].apply(lambda x: re.sub('\w*\d\w*','', x))#Убирает все цифры
df1["row"]=df1["row"].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))#Убирает пунктуацию (скобки, кавычки, и т.д.)

def clean_text(text):
    text = re.sub(r"\n", " ", text)
    return text
df1["row"] =df1["row"].apply(clean_text)

df1=df1[df1['row'].str.contains('[а-я]')]



train, test = train_test_split(df1, test_size=0.2,random_state=42)



print(tabulate(df1.sample(10),headers='keys'))
