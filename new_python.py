import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
import csv
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
import time
import os
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Reds):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

categories = ['prostitution', 'mezhnac', 'vera', 'politic', 'drugs', 'positive']
no_classes = len(categories)

data_path = 'data/data.csv'

tweet_tokenizer = TweetTokenizer()

tweet_data = []

with open(data_path) as f:
    reader = csv.reader(f, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    tweet_data = list(reader)


parsed_tweet = []

# stop words
stop = set(stopwords.words('russian'))

emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)

patt1 = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
patt2 = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')

for info in tweet_data:
    # delete links
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', info[0].lower(), flags=re.MULTILINE)
    # delete emojis
    text = emoji_pattern.sub(r'', text)  # no emoji
    text = patt1.sub(r'', text)  # no emoji
    text = patt2.sub(r'', text)  # no emoji

    # delete @
    # delete #
    l = " ".join(tweet_tokenizer.tokenize(text)).split(" ")
    filtered_sentence = [w for w in l if not w in stop and not w in string.punctuation
                         and (w[0] != '@' and w[0] != '#')]
    parsed_tweet.append(filtered_sentence)

# creates a corpus with each document having one string

for i in range(len(parsed_tweet)):
    parsed_tweet[i] = ' '.join(parsed_tweet[i])

# label the data

tweet_target = np.zeros(len(tweet_data))

labels = {
    'Предложение проституции': 0,
    'Разжигание межнациональной розни': 1,
    'Оскорбление чувств верующих': 2,
    'Посты политической направленности': 3,
    'Продажа наркотиков': 4,
    'positive': 5
}

labels_list = [key for key in labels]
labels_list_two = ['prostitution', 'mezhnac', 'vera', 'politic', 'drugs', 'positive']

for i in range(len(tweet_data)):
    tweet_target[i] = labels[tweet_data[i][1]]

X_train, X_test, train_y, test_y = train_test_split(parsed_tweet, tweet_target, test_size=0.2, random_state=42)

test_y_main = test_y

tfidf = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True)
train_x = tfidf.fit_transform(X_train)
test_x = tfidf.transform(X_test)

# transforming target classes into one-hot vectors
def vector_to_one_hot(vector,no_classes):
    vector=vector.astype(np.int32)
    m = np.zeros((vector.shape[0], no_classes))
    m[np.arange(vector.shape[0]), vector]=1
    return m

train_y = vector_to_one_hot(train_y,no_classes)
test_y = vector_to_one_hot(test_y, no_classes)

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 9
display_step = 100

beta = 1 # regularization parameter
# Network Parameters
n_hidden_1 = 10 # size of 1st hidden layer

num_input = train_x.shape[1] #input vector size
num_classes = no_classes

X = tf.placeholder("float", [None, num_input]) # place holder for nn input
Y = tf.placeholder("float", [None, num_classes]) # place holder for nn output

weights = {
    'h1' : tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1,num_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def neural_net (X):
    layer_1 = tf.add(tf.matmul(X, weights['h1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    out_layer = tf.add(tf.matmul(layer_1,weights['out']), biases['out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

logits = neural_net(X)
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))

loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate)

train_step = optimizer.minimize(loss)

#evaluate model
correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
y_pred = tf.argmax(logits,1)

saver = tf.train.Saver()

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

init = tf.global_variables_initializer()


def get_train_batch(batch_size, train_x, train_y):
    global train_index

    if train_index + batch_size >= train_x.shape[0]:
        train_index += batch_size
        return train_x[train_index:, :], train_y[train_index:, :]  # false to indicate no more training batches
    else:
        r = train_x[train_index:train_index + batch_size, :], train_y[train_index:train_index + batch_size, :]
        train_index += batch_size
        return r

with tf.Session() as sess:
    sess.run(init)

    tf.train.write_graph(sess.graph_def, '.', 'hellotensor.pbtxt')

    train_index = 0
    moreTrain = True
    step = 0
    while True:
        step += 1
        if train_index >= train_x.shape[0]:
            break
        batch_x, batch_y = get_train_batch(batch_size, train_x.todense(), train_y)
        if(not len(batch_y)):
            break
        sess.run(train_step, feed_dict={X: batch_x, Y: batch_y})

        if step % 10 == 0:
            cur_loss, cur_accuracy = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
            saver.save(sess, checkpoint_prefix, global_step=step)
            print('loss = %.2f , accuracy = %.2f , at step %d' % (cur_loss, cur_accuracy, step))

    print("done optimization")
    y_p = sess.run(y_pred, feed_dict={X: test_x.todense(),
                                      Y: test_y})
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: test_x.todense(),
                                        Y: test_y}))

    print("f1 score : ",
          metrics.f1_score(test_y_main, y_p, average=None))

    cnf_matrix = metrics.confusion_matrix(test_y_main, y_p)
    print(cnf_matrix)

    plt.plot()
    plot_confusion_matrix(cnf_matrix, labels_list_two)

    print(metrics.classification_report(test_y_main, y_p, target_names=labels_list_two))