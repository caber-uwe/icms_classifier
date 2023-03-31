import joblib
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import pandas as pd
import re
from random import random

max_features = 6000
Cat = joblib.load('Categories_ML.joblib')
Cat[-1] = 'xx.xx.xxx'
icms_dct = joblib.load('icms1_dictionnary.joblib') 
icms_dct['xx.xx.xxx'] = 'ICMS code not found \ ICMS code not found \ ICMS code not found'
accepted_words = joblib.load('accepted_words.joblib')
lst_stopwords = joblib.load('stopwords.joblib')

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    #remove if only two characters
    text_big = re.sub(r'\W*\b\w{1,2}\b', '', text) 
          
    ## Tokenize (convert from string to list)
    lst_text = text_big.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]
    
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
        ## removing tags
        ## removing digits
    ## back to string from list
    text = " ".join(lst_text)
    return text

def clean_text_data(df,col):
    # Separate data
    desc_lower = df[col]
    # Remove text before "|" character
    desc_split = desc_lower.str.split("|")
    desc_strip = desc_split.apply(lambda x: x[1] if len(x) > 1 else x[0])
    # Removing digits and words containing digits
    desc_nodigits = desc_strip.apply(lambda x: re.sub("\w*\d\w*", "", x))
    # Removing punctuation
    desc_nopunc = desc_nodigits.apply(lambda x: re.sub(r"[^\w\s]", "", x))
    # Removing additional whitespace
    desc_clean = desc_nopunc.apply(lambda x: re.sub(' +', ' ', x))
    return desc_clean

def goodCode(name,code,desc):
    A = {'success' : 'true' , "Message" : name ,'ICMS' : code ,"Description" : desc}
    R = A['ICMS'].split('.')
    D = [x.strip() for x  in A['Description'].split('\\')]
    A['R2'] = R[0]
    A['R3'] = R[1]
    A['R4'] = R[2]

    A['Desc2'] = D[0]
    A['Desc3'] = D[1]
    A['Desc4'] = D[2]
    return A

def predict(names,clf):
    if type(names) == str:
        names = [names]
    A = pd.DataFrame()
    A['comment_list'] = names
    A['comment_list'] = A.comment_list.apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords))
    A['comment_list_new'] = clean_text_data(A,'comment_list')
    vectorizer = CountVectorizer(max_features=max_features,strip_accents='ascii',vocabulary=accepted_words)
    X = vectorizer.fit_transform(A['comment_list_new']).toarray()
    #X = X.astype(dtype=bool).astype(dtype=int)
    y_pred = clf.predict(X)
    codes = [Cat[x] for x in y_pred]
    descs =[icms_dct[c] for c in codes]
    answer = [goodCode(names[i],codes[i],descs[i]) for i,_ in enumerate(names)]
    return answer