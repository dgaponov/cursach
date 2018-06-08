import numpy as np
import re
import itertools
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import csv
import string


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(data_path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.

    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    """

    tweet_data = []

    with open(data_path) as f:
        reader = csv.reader(f, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        tweet_data = list(reader)

    tweet_tokenizer = TweetTokenizer()
    parsed_tweet = []

    emoji_pattern = re.compile(
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)

    patt1 = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    patt2 = re.compile(
        u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')

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
        filtered_sentence = [w for w in l if not w in string.punctuation
                             and (w[0] != '@' and w[0] != '#')]
        parsed_tweet.append(filtered_sentence)

    # creates a corpus with each document having one string

    for i in range(len(parsed_tweet)):
        parsed_tweet[i] = ' '.join(parsed_tweet[i])

        # label the data

    tweet_target = []

    labels = {
        'Предложение проституции': [1, 0, 0, 0, 0],
        'Разжигание межнациональной розни': [0, 1, 0, 0, 0],
        'Оскорбление чувств верующих': [0, 0, 1, 0, 0],
        'Посты политической направленности': [0, 0, 0, 1, 0],
        'Продажа наркотиков': [0, 0, 0, 0, 1]
    }

    labels_list = [key for key in labels]
    labels_list_two = ['prostitution', 'mezhnac', 'vera', 'politic', 'drugs']

    for i in range(len(tweet_data)):
        tweet_target.append(labels[tweet_data[i][1]])

    return [parsed_tweet, np.array(tweet_target)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]