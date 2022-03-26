import random
import json
import pickle
import numpy as np

import nltk
#nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model

le = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot.h5')

def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [le.lemmatize(word) for word in sentence_words]
    return sentence_words

def word_bags(sentence):
    sentence_words = clean_up(sentence)
    # create a bag full of 0's
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def class_predict(sentence):
    word_bag = word_bags(sentence)
    predicted = model.predict(np.array([word_bag]))[0]
    ERROR_THRESHOLD = 0.25

    actual = [[i,r] for i,r in enumerate(predicted) if r>ERROR_THRESHOLD]

    actual.sort(key=lambda x: x[1], reverse=True)
    res_list = []
    for r in actual:
        res_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return res_list

def get_response(intents, intents_json):
    tag = intents[0]['intent']
    intents_list = intents_json['intents']
    for i in intents_list:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

'''
while True:
    msg = input("")
    ints = class_predict(msg)
    ans = get_response(ints, intents)
    print(ans)
if __name__ == "__chatbot__":
    main()
'''
