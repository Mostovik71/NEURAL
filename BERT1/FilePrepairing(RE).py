import pandas as pd
import re
import string
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
df=pd.read_excel('all ideas1.xlsx')
df1=df[['Описание идеи / Idea description','Parent Idea']]
df1["Описание идеи / Idea description"]=df1["Описание идеи / Idea description"].apply(lambda x: x.lower())#Переводит весь верхний регистр в нижний
df1["Описание идеи / Idea description"]=df1["Описание идеи / Idea description"].apply(lambda x: re.sub('\w*\d\w*','', x))#Убирает все цифры
df1["Описание идеи / Idea description"]=df1["Описание идеи / Idea description"].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))#Убирает пунктуацию (скобки, кавычки, и т.д.)
df1["Описание идеи / Idea description"]=df1["Описание идеи / Idea description"].apply(lambda x: re.sub('http\w*','', x))#Убирает все слова, начинающиеся с http (ссылки)
df1["Описание идеи / Idea description"]=df1["Описание идеи / Idea description"].apply(lambda x: re.sub('fujitsucom','', x))#Убирает все слова, начинающиеся с http (ссылки)
def clean_text(text):
    text = re.sub(r"\n", " ", text)
    return text
df1["Описание идеи / Idea description"] =df1["Описание идеи / Idea description"].apply(clean_text)

df1=df1[df1['Описание идеи / Idea description'].str.contains('[а-я]')]
#df1.to_csv('ideas2.csv',index=False,encoding='cp1251')


train, test = train_test_split(df1, test_size=0.2,random_state=42)
test.drop(['Parent Idea'],axis=1,inplace=True)

#df=pd.get_dummies(train,prefix='',columns=['Parent Idea'])
print(tabulate(df1.sample(10),headers='keys'))
