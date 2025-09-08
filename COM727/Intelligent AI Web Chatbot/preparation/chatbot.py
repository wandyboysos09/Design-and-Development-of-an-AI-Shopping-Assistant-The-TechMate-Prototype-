import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load intents file
intents = json.load(open('Preparation/intents.json'))

# Load trained model and data
words = pickle.load(open('Preparation/model/words.pkl', 'rb'))  # Vocabulary

classes = pickle.load(
    open('Preparation/model/classes.pkl', 'rb'))  # Intent labels

model = load_model('Preparation/model/chatbot_model.keras')  # Trained model


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize input
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]  # Normalize words
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)  # Clean sentence
    bag = [0] * len(words)  # Init bag
    for w in sentence_words:  # For each word in input
        for i, word in enumerate(words):  # Match against vocab
            if word == w:
                bag[i] = 1  # Mark presence
    return np.array(bag)  # Return BoW vector


def predict_class(sentence):
    bow = bag_of_words(sentence)  # Convert input to BoW

    # If no known words, return fallback
    if sum(bow) == 0:
        return [{'intent': 'fallback', 'probability': '1.0'}]

    res = model.predict(np.array([bow]), verbose=0)[0]  # Model prediction

    ERROR_THRESHOLD = 0.7  # Confidence threshold
    results = [[i, r] for i, r in enumerate(
        res) if r > ERROR_THRESHOLD]  # Filter intents
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by probability

    # If no intent passes threshold, return fallback
    if not results:
        return [{'intent': 'fallback', 'probability': '1.0'}]

    # Return predicted intents with probabilities
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']  # Get top intent

    # If fallback, return fallback response
    if tag == 'fallback':
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == 'fallback':
                return random.choice(i['responses'])

    # For other intents, return a random response
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

    # Final safeguard
    return "I'm not sure how to respond to that. Can you try asking something else?"


# Chat loop
print("Bot is ready! Type something to start chatting (type 'quit' to exit).")
print("I can help with information about our computer products, store hours, and more!")

while True:
    message = input("You: ")  # Get user input
    if message.lower() in ['quit', 'exit', 'bye', 'goodbye']:  # Exit commands
        print("Bot: Goodbye! Have a great day!")
        break

    if not message.strip():  # Handle empty input
        print("Bot: I didn't catch that. Could you please type something?")
        continue

    ints = predict_class(message)  # Predict intent
    res = get_response(ints, intents)  # Get response
    print("Bot:", res)  # Show response
