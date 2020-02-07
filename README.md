# Identify the Sentiments - Analytics Vidhya Contest

This project is submitted as python implementation in the contest of Analytics Vidhya called "Identify the Sentiments". I enjoyed the joining of this competition and all its process. This submited solution got the rank <b>147</b> in the public leaderboard.

The contest URL: https://datahack.analyticsvidhya.com/contest/linguipedia-codefest-natural-language-processing-1/

# Problem Statement
Sentiment analysis remains one of the key problems that has seen extensive application of natural language processing. This time around, given the tweets from customers about various tech firms who manufacture and sell mobiles, computers, laptops, etc, the task is to identify if the tweets have a negative sentiment towards such companies or products.

# Implementation Approach

## Dataset

- The train set contains 7,920 tweets
- The test set contains 1,953 tweets

## Text Cleaning and Preprocessing 
We applied the below text preposessing on the training and testing tweets sets:

- URLs removal: We have used Regular Expressions (or RegEx) to remove the URLs.
- Punctuation marks removal: remove any punction marks from the text.
- Numbers removal: replace any digits in the tweets with space.
- Whitespaces removal
- Convert the text to lowercase.
- Text normalization by reducing the words to its base form by using the <b>spacy</b> library.


## Tweets to ElMo Vectors

We imported and used the pretrained ELMo model from the Tensorflow Hub, where we extracted ELMo vectors for the cleaned tweets in the train and test datasets. Each tweet is represented by an ELMo vector of length 1024 interms of the tweet's words/tokens.


## Tweets to BERT Vectors

We imported and used the pretrained google BERT model, where we extracted BERT vectors for the cleaned tweets in the train and test datasets. Each tweet is represented by an BERT vector of length 768 interms of the tweet's words/tokens.


## Classifiaction Model Building and Evaluation

We have used the ELMo vectors and BERT vecyors as features of the train dataset to build and train a classification model. We evaluated our model by the F1 score metric since this is the official evaluation metric of the contest. We trained different classifications model as follows:

 - <b>BERT and ELMo verctors with Support vector machine model, the evaluation score for this SVM model is: <b>0.8899268298</b></b>


- ELMo vectors with Multi-layer Perceptron (MLP) Nueral Network Model, the evaluation score for this MLP model is: <b>0.881236842720449</b>
- ELMo vectors with Support vector machine model, the evaluation score for this SVM model is: <b>0.881868025014501</b>
- ELMo vectors with Simple Logistic Regression model, the evaluation score for this LR model is: <b>0.7761904761904763</b>


- BERT vectors with Multi-layer Perceptron (MLP) Nueral Network Model, the evaluation score for this MLP model is: <b>0.6096144315591238</b>
- BERT vectors with Support vector machine model, the evaluation score for this SVM model is: <b>0.8851479845833099</b>
- BERT vectors with Simple Logistic Regression model, the evaluation score for this LR model is: <b>0.8781415572832524</b>


## Future Work

- Try another classification models
- Try to use the word2vec vectors combined with the ELMo vectors as features.




