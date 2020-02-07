import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

from sklearn.metrics import f1_score
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

pd.set_option('display.max_colwidth', 200)

# read data
train = pd.read_csv("train_2kmZucJ.csv")
test = pd.read_csv("test_oJQbWVk.csv")

# load bert_train_new
pickle_in = open("bert_train_03032019.pickle", "rb")
bert_train_new = pickle.load(pickle_in)

# load bert_train_new
pickle_in = open("bert_test_03032019.pickle", "rb")
bert_test_new = pickle.load(pickle_in)

# load bert_train_new
pickle_in = open("elmo_train_03032019.pickle", "rb")
elmo_train_new = pickle.load(pickle_in)

# load bert_train_new
pickle_in = open("elmo_test_03032019.pickle", "rb")
elmo_test_new = pickle.load(pickle_in)


train_final =np.concatenate((bert_train_new,elmo_train_new),axis=1)
test_final = np.concatenate((bert_test_new,elmo_test_new),axis=1)

xtrain, xvalid, ytrain, yvalid = train_test_split(train_final, 
                                                  train['label'],  
                                                  random_state=42, 
                                                  test_size=0.2)


clf = svm.SVC(gamma='scale')

clf.fit(xtrain, ytrain)

preds_valid = clf.predict(xvalid)
print (f1_score(yvalid, preds_valid))

# make predictions on test set
preds_test = clf.predict(test_final)


# prepare submission dataframe
sub = pd.DataFrame({'id':test['id'], 'label':preds_test})

# write predictions to a CSV file
sub.to_csv("sub_svm.csv", index=False)

