# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
develop an LSTM-based model for recognizing the named entities in the text. Dataset - NER dataset
## Neural Network Model

Include the neural network model diagram.
![image](https://user-images.githubusercontent.com/75235128/199037523-16fc643c-44e5-4f83-88df-adcb694b925a.png)

## DESIGN STEPS

### STEP 1:
Import the necessary tensorflow modules

### STEP 2:
load the NER dataset

### STEP 3:
fit the model and then predict

## PROGRAM

```
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

data = pd.read_csv("ner_dataset.csv", encoding="latin1")

data = data.fillna(method="ffill")

sentences=data.groupby("Sentence #").apply(lambda s: [(w,p,t) for w,p,t in zip(s["Word"],s["POS"],s["Tag"])])

print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())

words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())

num_words = len(words)
num_tags = len(tags)

sentences[0]

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

word2idx

X1 = [[word2idx[w[0]] for w in s] for s in sentences]

X = sequence.pad_sequences(maxlen=50,
                  sequences=X1, padding="post",
                  value=num_words-1)

X

y1 = [[tag2idx[w[2]] for w in s] for s in sentences]

y1[0]

y = sequence.pad_sequences(maxlen=50,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)

y[0]

import tensorflow as tf

model=tf.keras.Sequential([
                     tf.keras.layers.Embedding(input_dim=40000,output_dim=300,input_length=50,embeddings_initializer='uniform'),
                     tf.keras.layers.Dense(num_tags,activation="softmax")
])

model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"])

history=model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test,y_test),
    batch_size=32, 
    epochs=3,
)

X_test[0]

i = 20
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))

pd.DataFrame(history.history).plot()
```

## OUTPUT
![image](https://user-images.githubusercontent.com/75235128/199038069-c449b319-ecd4-4b63-b6c4-238725053d2a.png)

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/75235128/199038159-b0b830a2-58dc-4722-83a3-67d4878332fd.png)

### Sample Text Prediction
![image](https://user-images.githubusercontent.com/75235128/199038290-f9934650-4039-46b6-9247-89f43cba8de4.png)

## RESULT
Thus an LSTM-based model for recognizing the named entities in the text is created and executed successfully.
