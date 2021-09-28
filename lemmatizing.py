from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    Doc
)
import pandas as pd
df2=pd.DataFrame([])
morph_vocab = MorphVocab()
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
df=pd.read_excel('file.xlsx')
for i, k in enumerate(df['Descr1']):
   doc=Doc(k)
   doc.segment(segmenter)
   doc.tag_morph(morph_tagger)
   tokens=[]
   for token in doc.tokens:
      token.lemmatize(morph_vocab)
      tokens.append(token.lemma)
   df2=df2.append(pd.concat([pd.Series(df[''].iloc[i]), pd.Series(str(" ".join(tokens)))], axis=1))
df2.to_excel('newfile.xlsx')
stop=1