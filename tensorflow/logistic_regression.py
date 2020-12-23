import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# MNIST dataset
num_classes = 10
num_features = 784  # 28 * 28

# Training params
l_rate = 0.01
epoches = 1000
batch_size = 256
display_step = 50

# Prepare MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Convert to float32
X_train, X_test = np.array(X_train, np.float32), np.array(X_test, np.float32)

# Flatten images to 1-D vector of 784 features (28 * 28)
X_train, X_test = X_train.reshape([-1, num_features]), X_test.reshape([-1, num_features])

# Normalize images values from [0, 255] to [0, 1]

X_train, X_test = X_train / 255., X_test / 255.

# Shuffle dataset
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# Weights of shape [784, 10], the 28 * 28 image features and total number of classes
W = tf.Variable(tf.ones([num_features, num_classes]), name='weights')

# Bias total number of classes
b = tf.Variable(tf.zeros([num_classes], name='bias'))


# Logistic regression ( Wx +b)
def logistic_reg(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)


# Cross-entropy loss function
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector
    y_true = tf.one_hot(y_true, depth=num_classes)

    # Clip prediction  values to avoid log(0) error
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)

    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), 1))


# Accuracy metric
def accuracy(y_pred, y_true):
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))

    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Stochastic gradient descent optimizer
optimizer = tf.optimizers.SGD(l_rate)


# Optimazation process
def run_optimizer(x, y):
    with tf.GradientTape() as g:
        pred = logistic_reg(x)
        loss = cross_entropy(pred, y)

    # Compute gradients
    gradients = g.gradient(loss, [W, b])

    # Update W and b following gradient
    optimizer.apply_gradients(zip(gradients, [W, b]))


# Run training for the given number of steps

for step, (batch_x, batch_y) in enumerate(train_data.take(epoches), 1):
    # Run the optimization to update W and b
    run_optimizer(batch_x, batch_y)

    if step % display_step == 0:
        pred = logistic_reg(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)

        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

pred = logistic_reg(X_test)
print("Test accuracy: %f" % (accuracy(pred, y_test)))

n_images = 5
test_images = X_test[:n_images]
predictsions = logistic_reg(test_images)

for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" %(np.argmax(predictsions.numpy()[i])))
