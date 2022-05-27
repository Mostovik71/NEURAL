def func(x):
    y = ' '.join(x)
    return y

morph = pymorphy2.MorphAnalyzer()
df=pd.read_excel('newwithoutstrange.xlsx')
patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")

def lemmatize(doc):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]

            tokens.append(token)

    return tokens
df = df.dropna().drop_duplicates()
data1 = df['Descr1'].apply(lemmatize)
data2 = df['Descr2'].apply(lemmatize)
