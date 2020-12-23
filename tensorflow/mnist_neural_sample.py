import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Model, layers
from tensorflow.keras.datasets import mnist

# MNIST dataset params
n_classes = 10
n_features = 784

# Training params
l_rate = 0.01
epoches = 1000
batch_size = 256
display_steps = 100

# Network param
n_hidden_1 = 128
n_hidden_2 = 256

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Convert to float32
X_train, X_test = np.array(X_train, np.float32), np.array(X_test, np.float32)

# Flatten images to 1-D array
X_train, X_test = X_train.reshape([-1, n_features]), X_test.reshape([-1, n_features])

# Normalize images values from [0, 255] to [0,1]
X_train, X_test = X_train / 255., X_test / 255.

# Shuffle data using tensorflow tf.data
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


# Create TF Model

class NeuralNet(Model):

    def __init__(self):
        super(NeuralNet, self).__init__()

        # First fully-connected hidden layer
        self.fc1 = layers.Dense(n_hidden_1, activation=tf.nn.relu)

        # Second fully connected hidden layer
        self.fc2 = layers.Dense(n_hidden_2, activation=tf.nn.relu)

        # Output layer
        self.out = layers.Dense(n_classes)

    # set forward pass
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)

        if not is_training:
            x = tf.nn.softmax(x)

        return x


# Build network model
neural_net = NeuralNet()


# Cross-entropy
def cross_entrpoy(x, y):
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits compute cross-entropy
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)


# Accuracy metric
def accuracy(y_pred, y_true):
    curr_pred = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))

    return tf.reduce_mean(tf.cast(curr_pred, tf.float32), axis=-1)


# Stochastic gradient descent
optimizer = tf.optimizers.SGD(l_rate)


# Optimazation process
def run_optimazatio(x, y):
    with tf.GradientTape() as g:
        pred = neural_net(x, is_training=True)

        loss = cross_entrpoy(pred, y)

    # Variable update i.e trainable  variables
    trainable_vars = neural_net.trainable_variables

    # Compute gradiennts
    gradients = g.gradient(loss, trainable_vars)
    # Update W and b following gradients
    optimizer.apply_gradients(zip(gradients, trainable_vars))


# Run training for  the given steps
for step, (batch_x, batch_y) in enumerate(train_data.take(epoches), 1):
    run_optimazatio(batch_x, batch_y)

    if step % display_steps == 0:
        pred = neural_net(batch_x, is_training=True)
        loss = cross_entrpoy(pred, batch_y)

        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

pred = neural_net(X_test, is_training=False)
print("Accuracy: %f" % accuracy(pred, y_test))

n_images = 5
test_images = X_test[:n_images]

predictions = neural_net(test_images)

for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()

    print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))
