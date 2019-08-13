import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
#from sklearn import svm
from sklearn.metrics import f1_score
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

pd.set_option('display.max_colwidth', 200)

# read data
train = pd.read_csv("train_2kmZucJ.csv")
test = pd.read_csv("test_oJQbWVk.csv")

# load elmo_train_new
pickle_in = open("elmo_train_03032019.pickle", "rb")
elmo_train_new = pickle.load(pickle_in)

# load elmo_train_new
pickle_in = open("elmo_test_03032019.pickle", "rb")
elmo_test_new = pickle.load(pickle_in)


xtrain, xvalid, ytrain, yvalid = train_test_split(elmo_train_new, 
                                                  train['label'],  
                                                  random_state=42, 
                                                  test_size=0.2)


clf = MLPClassifier(activation='logistic', solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(100,50), random_state=1,
                    learning_rate_init=0.1,max_iter=500)

clf.fit(xtrain, ytrain)

preds_valid = clf.predict(xvalid)
print (f1_score(yvalid, preds_valid))

# make predictions on test set
preds_test = clf.predict(elmo_test_new)


# prepare submission dataframe
sub = pd.DataFrame({'id':test['id'], 'label':preds_test})

# write predictions to a CSV file
sub.to_csv("sub_mlp.csv", index=False)

