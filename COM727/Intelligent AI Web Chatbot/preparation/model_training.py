from tensorflow.keras.callbacks import EarlyStopping
import random
import json
import pickle  # For saving processed data
import numpy as np
import nltk  # For text processing
from nltk.stem import WordNetLemmatizer  # For word lemmatization
from tensorflow.keras.models import Sequential  # For building NN model
from tensorflow.keras.layers import Dense, Activation, Dropout  # Model layers
from tensorflow.keras.optimizers import SGD  # Optimizer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress warnings

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()  # used to reduce vocabulary size; running -> run

intents = json.load(open('Preparation/intents.json'))  # Load intents file

words = []  # Store all words
classes = []  # Store intent classes
documents = []  # Store patterns with their tags

# Ignore chars
ignore_letters = ['?', '!', '.', ',', "'", '"', '-', '_',
                  '(', ')', '[', ']', '{', '}', ';', ':', '/', '\\', '@', '#', '$', '%', '^', '&', '*', '+', '=']

# Generic fallback patterns
fallback_patterns = [
    "football", "sports", "music", "movies", "weather", "politics",
    "news", "celebrity", "gossip", "travel", "vacation", "holiday",
    "recipe", "cooking", "restaurant", "car", "vehicle", "insurance",
    "health", "fitness", "exercise", "relationship", "family", "friends"
]

# Process intents
for intent in intents['intents']:  # Loop through intents
    if intent['tag'] == 'fallback':  # Add extra patterns to fallback
        intent['patterns'].extend(fallback_patterns)

    for pattern in intent['patterns']:  # Loop through patterns
        word_list = nltk.word_tokenize(pattern)  # Tokenize pattern
        words.extend(word_list)  # Add to words
        documents.append((word_list, intent['tag']))  # Add doc pair
        if intent['tag'] not in classes:  # Add unique tag
            classes.append(intent['tag'])

# Clean and lemmatize words
words = [lemmatizer.lemmatize(word.lower())
         for word in words if word not in ignore_letters]
words = sorted(set(words))  # Sort unique words

classes = sorted(set(classes))  # Sort unique classes

# Save processed data
pickle.dump(words, open('Preparation/model/words.pkl', 'wb'))

# Save classes
pickle.dump(classes, open('Preparation/model/classes.pkl', 'wb'))

# Training data
training = []  # Feature + Label pairs
output_empty = [0] * len(classes)  # Empty label template

for document in documents:  # Loop through docs
    bag = []  # Bag of words
    word_patterns = [lemmatizer.lemmatize(
        word.lower()) for word in document[0]]  # Lemmatized tokens

    for word in words:  # Build bag
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)  # Copy empty output
    output_row[classes.index(document[1])] = 1  # Mark intent
    training.append([bag, output_row])  # Add training pair

random.shuffle(training)  # Shuffle training data

# Split into features/labels
train_x = []
train_y = []
for bag, output_row in training:
    train_x.append(bag)  # Features
    train_y.append(output_row)  # Labels

train_x = np.array(train_x)  # Convert to array
train_y = np.array(train_y)

# It's a feedforward neural network (a simple “vanilla” deep learning model).
# Its job: take an input sentence (turned into a bag-of-words vector) and
# decide which intent it belongs to (like "greeting", "order_status", "fallback").

# Building the model
model = Sequential()

# relu activation keeps positive value drops negative one
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))

# randomly turning off 50% of neurons preventing from memorizing from training data (overfitting)
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))  # Hidden layer
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))  # Hidden layer
model.add(Dropout(0.5))
# softmax makes the output upto 1.0
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6,
          # sgd tweeks the weight a little each step.
          momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])


# if the training doesn't improve for 200 epochs it will stop
early_stop = EarlyStopping(monitor='loss', patience=200, verbose=1)

# Train model
hist = model.fit(train_x, train_y, epochs=1000, batch_size=16,
                 verbose=1, callbacks=[early_stop])  # Train with ES
model.save('Preparation/model/chatbot_model.keras')  # Save model

print("Model training completed!")
