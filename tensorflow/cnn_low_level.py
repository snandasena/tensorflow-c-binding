import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Model, layers
from tensorflow.keras.datasets import mnist

# MNIST dataset params
n_classes = 10
n_features = 784

# Training params
l_rate = 0.001
epoches = 200
batch_size = 128
display_steps = 10

# Network params
conv1_filters = 32
conv2_filters = 64
fc1_units = 1024

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


# Create a some wrapper
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)


def maxpoll2d(x, k=2):
    # Maxpool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Store layers weigths and bias
random_normal = tf.initializers.RandomNormal()

weights = {
    # Conv layer 1: 5x5 conv, 1 input, 32 filters
    'wc1': tf.Variable(random_normal([5, 5, 1, conv1_filters])),
    # Conv layer 2: 5x5 conv, 32 inouts, 64 filters
    'wc2': tf.Variable(random_normal([5, 5, conv1_filters, conv2_filters])),
    # Fully connected layer: 7x7x64 inputs, 1024 units
    'wd1': tf.Variable(random_normal([7 * 7 * 64, fc1_units])),
    # FC output layer: 1024 inputs, 10 units
    'out': tf.Variable(random_normal([fc1_units, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.zeros([conv1_filters])),
    'bc2': tf.Variable(tf.zeros([conv2_filters])),
    'bd1': tf.Variable(tf.zeros([fc1_units])),
    'out': tf.Variable(tf.zeros([n_classes]))
}


# Create model
def conv_net(x):
    # Input shape: [-1, 28,28,1]
    x = tf.reshape(x, [-1, 28, 28, 1])

    # Convolution layer. Output shape: [-1, 28, 28, 32]
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    # Max pooling(down sampling). output shape [-1, 14,14,32]
    conv1 = maxpoll2d(conv1, k=2)

    # Convolution layer: Output shape: [-1, 14,14, 64]
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # Max pooling. output shape [-1, 7, 7, 64]
    conv2 = maxpoll2d(conv2, k=2)

    # Reshape conv2 output to fit fully connected layer input, Output  shape: [-1, 7*7*64]
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    # Fully connect layer, Output shape: [ -1, 1024]
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])

    # Apply ReLU to fc1 output for non-linearity
    fc1 = tf.nn.relu(fc1)

    # Fully connected layer, output shape [-1, 10]
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    # Apply softmax to normalize the logists to a probability distribution

    return tf.nn.softmax(out)


# Cross-entropy
def cross_entropy(y_pred, y_true):
    # Encode labels to a one hot vector
    y_true = tf.one_hot(y_true, depth=n_classes)

    # Clip prediction values to avoid log(0) error
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)

    # Compute cross-entropy
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# ADAM optimizer
optimizer = tf.optimizers.Adam(l_rate)


# Optimization process
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automate differentiation

    with tf.GradientTape() as g:
        pred = conv_net(x)
        loss = cross_entropy(pred, y)

    # Variables to update, i.e trainable vars
    trainable_vars = list(weights.values()) + list(biases.values())

    # Compute gradients
    gradients = g.gradient(loss, trainable_vars)

    # Update W and b following gradient
    optimizer.apply_gradients(zip(gradients, trainable_vars))


# Run training for the given number of steps
for step, (batch_x, batch_y) in enumerate(train_data.take(epoches), 1):
    # Run the optimization to update   W and b values
    run_optimization(batch_x, batch_y)

    if step % display_steps == 0:
        pred = conv_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)

        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# Test  model on validation set
pred = conv_net(X_test)
print("Test accuracy: %f" % accuracy(pred, y_test))

# Predict 5 images from validation set.
n_images = 5
test_images = X_test[:n_images]
predictions = conv_net(test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))
