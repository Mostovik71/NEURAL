import nltk
import pandas as pd
import sklearn
from tabulate import tabulate
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')#список стоп-слов
#string=input("Введите строку: ")#Ввод строки
f = open("test.txt", 'r')
fromf=f.read()
strarray=nltk.sent_tokenize(fromf)#Список с предложениями текста
#vectorizer = CountVectorizer()#1 или 0, встречается в предложении слово или нет
vectorizer=TfidfVectorizer()#TF-IDF оценка, "полезность" слова
X = vectorizer.fit_transform(strarray)
print(strarray)
print(set(stopwords.words('english')))
print(vectorizer.get_feature_names())
features=vectorizer.get_feature_names()
df1=pd.DataFrame(features)#датафрейм из заголовков
df2=pd.DataFrame(X.toarray())#Количество слов в тексте
df3=pd.concat([df1,df2],axis=1)
df2.columns=features
print(tabulate(df2,headers='keys',tablefmt='psql'))
