import pandas as pd
import numpy as np
import spacy
import re
import pickle
import logging
from bert_serving.client import BertClient

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

pd.set_option('display.max_colwidth', 200)

# read data
train = pd.read_csv("train_2kmZucJ.csv")
test = pd.read_csv("test_oJQbWVk.csv")

print (train.shape, test.shape)
print (train['label'].value_counts())
print (train.head())

# data cleaning: remove URL's from train and test
train['clean_tweet'] = train['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))
test['clean_tweet'] = test['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))

# remove twitter handles (@user)
train['clean_tweet'] = train['clean_tweet'].apply(lambda x: re.sub("@[\w]*", '', x))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: re.sub("@[\w]*", '', x))
  
# remove punctuation marks
punctuation = '.,\'!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

train['clean_tweet'] = train['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

# convert text to lowercase
train['clean_tweet'] = train['clean_tweet'].str.lower()
test['clean_tweet'] = test['clean_tweet'].str.lower()

# remove numbers
train['clean_tweet'] = train['clean_tweet'].str.replace("[0-9]", " ")
test['clean_tweet'] = test['clean_tweet'].str.replace("[0-9]", " ")

# remove whitespaces
train['clean_tweet'] = train['clean_tweet'].apply(lambda x:' '.join(x.split()))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ' '.join(x.split()))

#Normalize the words to its base form
# import spaCy's language model
nlp = spacy.load('en', disable=['parser', 'ner'])

# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

#train['clean_tweet'] = lemmatization(train['clean_tweet'])
#test['clean_tweet'] = lemmatization(test['clean_tweet'])

# Extract BERT embeddings function
def bert_vectors(x):
  
    # make a connection with the BERT server using it's ip address
    bc = BertClient()
    
    return bc.encode(x.tolist())

# Extract BERT embeddings
bert_train = bert_vectors(train['clean_tweet'])
bert_test = bert_vectors(test['clean_tweet'])

# save bert_train_new
pickle_out = open("bert_train_03032019.pickle","wb")
pickle.dump(bert_train, pickle_out)
pickle_out.close()

# save bert_test_new
pickle_out = open("bert_test_03032019.pickle","wb")
pickle.dump(bert_test, pickle_out)
pickle_out.close()
