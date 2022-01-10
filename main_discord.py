import numpy as np
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import random
import discord
import sys

stemmer = LancasterStemmer()

def one_hot_encode_tokenized(tokenized_sentence):
    one_hot_encoded_sentence = np.zeros(len(words))
    for i, word in enumerate(words):
        if word in tokenized_sentence:
            one_hot_encoded_sentence[i] = 1

    return one_hot_encoded_sentence

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.02:
            self.model.stop_training = True


def one_hot_encode(sentence):
    tokenized_sentence = nltk.word_tokenize(sentence)
    tokenized_sentence = [stemmer.stem(w.lower()) for w in tokenized_sentence]
    return np.array([one_hot_encode_tokenized(tokenized_sentence)])


dataFile = "./data.json"
with open(dataFile, 'r') as f:
    jsonData = json.load(f)

words = []
classes = []
training_data = []
training_class = []
output_options = []

for intent in jsonData['intents']:
    output_options.append(intent['responses'])
    for sentence in intent['pattern']:
        current_words = nltk.word_tokenize(sentence)
        current_words = [stemmer.stem(w.lower()) for w in current_words]

        words.extend(current_words)
        training_data.append(current_words)
        training_class.append(intent['class'])

    if intent['class'] not in classes:
        classes.append(intent['class'])


words = list(set(words))

output_labels = []
one_hot_encoded_training_data = []

for cl in training_class:
    output_labels.append(np.zeros(len(classes)))
    class_index = classes.index(cl)
    output_labels[-1][class_index] = 1

for example in training_data:
    one_hot_encoded_training_data.append(one_hot_encode_tokenized(example))


output_labels = np.array(output_labels)
one_hot_encoded_training_data = np.array(one_hot_encoded_training_data)

wordCount = len(words)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(wordCount, input_shape = [wordCount]),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(output_labels[0]), activation="softmax")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callback = myCallback()
model.fit(one_hot_encoded_training_data, output_labels, epochs = 55, callbacks=[callback])

BOT_TOKEN = "OTI3ODQyMjg4NzI0NDk2NDA0.YdQGeA._258GNuP8ausH7o8M4W4xUiZGFU"

client = discord.Client()

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('%nbot'):
        sentence = message.content[6:]
        probabilities = model.predict(one_hot_encode(sentence))[0]
        bestPredictionIndex = np.argmax(probabilities)
        if probabilities[bestPredictionIndex] > 0.5:
            await message.channel.send(random.choice(output_options[bestPredictionIndex]))
        else:
            await message.channel.send("I do not understand your inquiry young master!")


def talk(model, output_options):
    flag = True
    print("Chat with my chatbot. Write quit to exit")

    while flag:
        sentence = input("You: ")
        if sentence == "quit":
            print("Goodbye")
            flag = 0
        else:
            probabilities = model.predict(one_hot_encode(sentence))[0]
            bestPredictionIndex = np.argmax(probabilities)
            if probabilities[bestPredictionIndex] > 0.5:
                print(random.choice(output_options[bestPredictionIndex]))
            else:
                print("I do not understand your inquiry young master!")


if len(sys.argv) > 1 and sys.argv[1] == "discord":
    print("Connecting to discord")
    client.run(BOT_TOKEN)
else:
    talk(model, output_options)

