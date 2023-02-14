import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow as tf 
import random
import nltk
import json

stemmer = LancasterStemmer()
nltk.download('punkt')
words = []
words_labels = []
docs_x = []
docs_y = []

# Récuperee data dans le fichier
with open('messages.json') as file:
    data = json.load(file)

# algo pour training
for mess in data['messages']:
    for inputW in mess['inputWords']:
        wrds = nltk.word_tokenize(inputW)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(mess["label"])

    if mess['label'] not in words_labels:
        words_labels.append(mess['label'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

words_labels = sorted(words_labels)

training = []
output = []

out_empty = [0 for _ in range(len(words_labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[words_labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

# Start training model
tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


def wordsContainer(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def lauchConversation():
    print("Commencer la conversation (exit pour arrêter)!")
    while True:
        inp = input("Vous: ")
        if inp.lower() == "exit":
            break

        results = model.predict([wordsContainer(inp, words)])
        results_index = numpy.argmax(results)
        label = words_labels[results_index]

        for tg in data["messages"]:
            if tg['label'] == label:
                responses = tg['responses']

        # Print random answer in the list
        print(random.choice(responses))



lauchConversation()