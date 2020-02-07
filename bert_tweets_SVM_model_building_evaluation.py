import pickle
import pandas as pd
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


xtrain, xvalid, ytrain, yvalid = train_test_split(bert_train_new, 
                                                  train['label'],  
                                                  random_state=42, 
                                                  test_size=0.2)


clf = svm.SVC(gamma='scale')

clf.fit(xtrain, ytrain)

preds_valid = clf.predict(xvalid)
print (f1_score(yvalid, preds_valid))

# make predictions on test set
preds_test = clf.predict(bert_test_new)


# prepare submission dataframe
sub = pd.DataFrame({'id':test['id'], 'label':preds_test})

# write predictions to a CSV file
sub.to_csv("sub_svm.csv", index=False)

