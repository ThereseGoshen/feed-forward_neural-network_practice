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

df = pd.read_csv("spam.csv", encoding='latin-1')

#print(df.head())

# only keep the 'v1' and 'v2' columns
df = df[['v1', 'v2']]

# rename the columns to 'label' and 'text'
df.columns = ['label', 'text']

# map 'ham' to 0 and 'spam' to 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# print(df.head())
# print(df['label'].value_counts())
# looks like there is alot more ham than spam

# split and vectorize the data
x_train, x_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state = 42
)

# print(x_train.shape)
vectorize = TfidfVectorizer(max_features=2000, stop_words='english')
x_train_vec = vectorize.fit_transform(x_train)
x_test_vec = vectorize.transform(x_test)

# dont forget the square brackets to make it a 2D array
model = keras.Sequential([
    layers.Input(shape=(x_train_vec.shape[1],)),
    layers.Dense(100, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train = model.fit(
    x_train_vec.toarray(), # array
    y_train, # labels
    validation_data=(x_test_vec.toarray(), y_test), # validation to eval after each
    epochs=3, # the iterations
    batch_size=10  
)

y_pred_probs = model.predict(x_test_vec.toarray()).flatten()
y_pred = [1 if p >= 0.5 else 0 for p in y_pred_probs]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
