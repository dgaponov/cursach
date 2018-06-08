from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics


remove = ()

data_train = fetch_20newsgroups(subset='train',
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test',
                               shuffle=True, random_state=42,
                               remove=remove)

target_names = data_train.target_names
y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a sparse vectorizer")
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)

print("n_samples: %d, n_features: %d" % X_train.shape)

print("Extracting features from the test data using the same vectorizer")
X_test = vectorizer.transform(data_test.data)
print("n_samples: %d, n_features: %d" % X_test.shape)

#X_train = X_train.toarray()
#X_test = X_test.toarray()

#print(str(X_train))
print("feature size train: " + str(X_train[0].size))
print("feature size test: " + str(X_test[0].size))

print("number of classes: " + str(len(target_names)))

# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[X_train[0].size])]


# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[50, 500, 50],
                                        n_classes=len(target_names))

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(X_train)},
      y=np.array(y_train),
      num_epochs=None,
      shuffle=True)


# Train model.
classifier.train(input_fn=train_input_fn, steps=2000)

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(X_test)},
      y=np.array(y_test),
      num_epochs=1,
      shuffle=False)

predictions = list(classifier.predict(input_fn=test_input_fn))
predicted_classes = [p["classes"] for p in predictions]

predicted_classes = np.array(predicted_classes, dtype=np.int32).flatten()

score = metrics.accuracy_score(y_test, predicted_classes)
print(score)
print("accuracy:  %%%0.1f" % (score * 100))
