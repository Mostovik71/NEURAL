import nltk
import pandas as pd
import sklearn
from tabulate import tabulate
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords
from tabulate import tabulate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
nltk.download('punkt')
nltk.download('stopwords')#список стоп-слов
#string=input("Введите строку: ")#Ввод строки
#f = open("test.txt", 'r')
#fromf=f.read()
train=pd.read_csv('trainq.csv',nrows=15000)
train.dropna(inplace=True)
tfidf = TfidfVectorizer(analyzer = 'word',
                        stop_words = 'english',
                        lowercase = True,
                        max_features = 100000,
                        norm = 'l1')
BagOfWords = pd.concat([train.question1, train.question2], axis = 0)
tfidf.fit(BagOfWords)
train_q1_tfidf = tfidf.transform(train.question1)
train_q2_tfidf = tfidf.transform(train.question2)
X = abs(train_q1_tfidf - train_q2_tfidf)
y = train['is_duplicate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Логистическая регрессия дает score = 0.66(100000 features)
'''
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
df1=pd.DataFrame(y_test)
df2=pd.DataFrame(y_pred)
df1.reset_index(drop=True, inplace=True)
df=pd.concat([df1,df2],axis=1)
columns=['true','model']
df.columns=columns
print(lr.score(X_test,y_test))
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(lr,X_test, y_test)
plt.show()
'''
#Random Forest(1000 деревьев - 0.665, 500 - 0.666)
rf = RandomForestClassifier(n_estimators = 500,
                            min_samples_leaf = 10,
                            n_jobs = -1)
rf.fit(X_train, y_train)

sns.set(font_scale=1.5)
sns.set_color_codes("muted")

plt.figure(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1], pos_label=1)
lw = 2
plt.plot(fpr, tpr, lw=lw, label='ROC curve ')
plt.plot([0, 1], [0, 1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

'''
strarray=nltk.sent_tokenize('my name is vlad and my name is')#Список с предложениями текста
vectorizer = CountVectorizer()#1 или 0, встречается в предложении слово или нет
#vectorizer=TfidfVectorizer()#TF-IDF оценка, "полезность" слова
X = vectorizer.fit_transform(strarray)
features=vectorizer.get_feature_names()
print(strarray)
#print(set(stopwords.words('english')))
print(features)

df1=pd.DataFrame(features)#датафрейм из заголовков
df2=pd.DataFrame(X.toarray())#Количество слов в тексте
df3=pd.concat([df1,df2],axis=1)
df2.columns=features
print(tabulate(df2,headers='keys',tablefmt='psql'))
'''