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


# Create TF model
class ConvNet(Model):

    def __init__(self):
        super(ConvNet, self).__init__()

        # Convolutional layer with 32 filters and a kernel size of 5
        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu)

        # Max pooling with kernel size 2 and strides of 2
        self.maxpool1 = layers.MaxPool2D(2, strides=2)

        # Convulutional layer with 64 filters and a kernel size of 3
        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)

        # Max pooling with  kernel size 2 and strides of 2
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        # Flatten data  to a 1-D vector for the fully connected layer
        self.flatten = layers.Flatten()

        # Fully connected layer
        self.fc1 = layers.Dense(1024)

        # Apply dropout
        self.dropout = layers.Dropout(rate=0.5)

        # Output layer, class prediction
        self.out = layers.Dense(n_classes)

    # Set forward pass
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)

        if is_training:
            x = tf.nn.softmax(x)

        return x


# Build neural network model
neural_net = ConvNet()


# Cross-entropy loss
def cross_entropy_loss(x, y):
    # Convert labels to int 64 for cross entropy loss
    y = tf.cast(y, tf.int64)

    # Apply softmax to logits and compute cross-entropy
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)

    # Average the loss across the batch
    return tf.reduce_mean(loss)


# Accuracy metric
def accuracy(y_pred, y_true):
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))

    return tf.reduce_mean(tf.cast(correct_pred, tf.float32), axis=-1)


# Stochastic gradient descent optimizer
optimizer = tf.optimizers.Adam(l_rate)


# Run optimazation process
def run_optimation(x, y):
    with tf.GradientTape() as g:
        pred = neural_net(x, is_training=True)
        loss = cross_entropy_loss(pred, y)

    # Variable update
    trainable_vars = neural_net.trainable_variables
    # Compute gradients
    gradients = g.gradient(loss, trainable_vars)

    # Update W and b following gradients
    optimizer.apply_gradients(zip(gradients, trainable_vars))


# Run training for the given number of steps
for step, (batch_x, batch_y) in enumerate(train_data.take(epoches), 1):

    # Run optimaztion update
    run_optimation(batch_x, batch_y)

    if step % display_steps == 0:
        pred = neural_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)

        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# Test model accuracy
pred = neural_net(X_test)
print("Test accuracy: %f" % accuracy(pred, y_test))

n_images = 5
test_images = X_test[:n_images]
predictions = neural_net(test_images)

for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()

    print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))
