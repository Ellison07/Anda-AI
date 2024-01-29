import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

from flask import Flask, request
import requests
from twilio.twiml.messaging_response import MessagingResponse

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents/firstAidIntents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('models/chatbotmodel4.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)
                      for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res)
               if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]],
                            'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("KAPPU BOT IS RUNNING !")


app = Flask(__name__)


@app.route('/bot', methods=['POST'])
def bot():
    user_msg = request.values.get('Body', '')
    bot_resp = MessagingResponse()
    msg = bot_resp.message()
    ints = predict_class(user_msg)
    res = get_response(ints, intents)
    msg.body(res)
    if 'quote' in user_msg:
        r = requests.get('https://api.quotable.io/random')
        if r.status_code == 200:
            data = r.json()
            quote = f'{data["content"]} ({data["author"]})'
        else:
            quote = 'SORRY'
        msg.body(quote)
    print(str(user_msg) + " -> " + str(res))
    return str(bot_resp)


if __name__ == '__main__':
    app.run(debug=True)