import numpy as np
import pandas as pd
import nltk 
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l2, l1, l1_l2
from keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

truen = pd.read_csv("TrueNews-1.csv")
falsen = pd.read_csv("FakeNews-2.csv")

#create a column for labels of true and false
truen["Label"] = 1
falsen["Label"] = 0

#print (truen.head())
#print (falsen.head())

#combine the two datasets
news = pd.concat([truen, falsen])

#making the data all lowercase
#replacing the Reuters so it's not a determining factor
news['Title'] = news['Title'].str.lower()
news['Text'] = news['Text'].str.lower()

news['Title'] = news['Title'].str.replace("reuters", "", regex=False)
news['Text'] = news['Text'].str.replace("reuters", "", regex=False)

# shuffle the dataset
# frac is the fraction of rows to return
# reset_index(drop=True) to reset the index after shuffling

news = news.sample(frac=1, random_state=42).reset_index(drop=True) #seed the random

# title and text column are combined for input to train_test_split
news['Combined'] = news['Title'] + " " + news['Text']

#test size 0.2 means 80% training, 20% testing
#random state is the seed
x_train, x_test, y_train, y_test = train_test_split(
    news['Combined'], news['Label'], test_size=0.2, random_state = 42
)

# this prints the shape of the training data
# print(x_train.shape)

# do we have a balanced dataset?
print(news['Label'].value_counts())


# convert the text to TF-IDF vectors 
# max_features = how many top words to keep
# stop_words = 'english'
# ngram_range = (1, 1) for unigrams (which are single words)
# max_df and min_df to remove rare and common words

vectorize = TfidfVectorizer(max_features=5000, stop_words='english')
x_train_vec = vectorize.fit_transform(x_train)
x_test_vec = vectorize.transform(x_test)

# build a feed forward model

# layers is a list of layers in the model
# each layer is added to the model in the order (relu, sigmoid)
# shape of input layer is the number of features (5000)
# the Dense(100) is the node count in the hidden layer
# the output layer has 1 node for binary classification (true/false)

model = keras.Sequential([
    layers.Input(shape=(x_train_vec.shape[1],)),
    layers.Dense(100, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# adam is optimizer
# binary_crossentropy is the loss function for binary classification
# accuracy is the metric to evaluate the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# now train the model
# epochs is the number of times to iterate over the training data
# batch_size is the number of samples per gradient update
train = model.fit(
    x_train_vec.toarray(), # array 
    y_train, # labels 
    validation_data=(x_test_vec.toarray(), y_test), # validation to eval after each epoch
    epochs=3, # the iterations
    batch_size=10  # the batch size is the number of samples per gradient update
    )

# evaluate the model
loss, accuracy = model.evaluate(x_test_vec.toarray(), y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# get predictions as probabilities
pred_probs = model.predict(x_test_vec.toarray()).flatten()

# convert probabilities to binary predictions (1 for True, 0 for False)
y_pred = [1 if p >= 0.5 else 0 for p in pred_probs]

# confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])

# print precision, recall, and F1 per category
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["True", "False"]))