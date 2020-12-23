import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# MNIST dataset params
n_classes = 10
n_features = 784

# Training params
l_rate = 0.001
epoches = 3000
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

# random value generator to initialize weights
random_normal = tf.initializers.RandomNormal()

# weights
weights = {
    'h1': tf.Variable(random_normal([n_features, n_hidden_1])),
    'h2': tf.Variable(random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([n_classes]))
}


# Create model

def neural_net(x):
    # Hidden fullu connected layer with 128 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])

    # Apply sigmoid to layer_1 output for non-linearity
    layer_1 = tf.nn.sigmoid(layer_1)

    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    # Apply sigmoid to layer_2 output for non-linearity
    layer_2 = tf.nn.sigmoid(layer_2)

    # Output fully connected layer with  a neuron for each class
    output_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    # Apply softmax to nomalize the logits to a probabilty distribution
    return tf.nn.softmax(output_layer)


# Cross-entroy
def cross_entry(y_pred, y_true):
    # encode label to a one hot vector
    y_true = tf.one_hot(y_true, depth=n_classes)

    # clip prediction values to avoid log(0) errors
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)

    # Compute cross-entrpy

    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Stochastic gradient descent optimizer
optimizer = tf.optimizers.SGD(l_rate)


# Optimizer process
def run_optimizer(x, y):
    with tf.GradientTape() as g:
        pred = neural_net(x)
        loss = cross_entry(pred, y)

    # Variables to update
    trainable_vars = list(weights.values()) + list(biases.values())

    # Compute gradient
    gradients = g.gradient(loss, trainable_vars)
    # Update W and b following gradients
    optimizer.apply_gradients(zip(gradients, trainable_vars))


# Run training for the given steps
for step, (batch_x, batch_y) in enumerate(train_data.take(epoches), 1):

    run_optimizer(batch_x, batch_y)
    if step % display_steps == 0:
        pred = neural_net(batch_x)
        loss = cross_entry(pred, batch_y)
        acc = accuracy(pred, batch_y)

        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# Test model on validation set
pred = neural_net(X_test)
print("Test accuracy: %f" % accuracy(pred, y_test))

n_images = 5
test_images = X_test[:n_images]
predictions = neural_net(test_images)

for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()

    print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))
