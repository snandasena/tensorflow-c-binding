import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

rnd = np.random

# Parameters
l_rate = 0.01
epoches = 1000
display_rate = 50

# Training data
X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
              7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
              2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

# Weights and bias initialization reandomly

W = tf.Variable(rnd.randn(), name='weights')
b = tf.Variable(rnd.randn(), name='bais')


# Linear regression - > y = Wx + b

def liner_reg(x):
    return W * x + b


# Mean square error
def mean_error(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))


# Stochastic gradient descent optimizer
optmizer = tf.optimizers.SGD(l_rate)


# Optimizer process
def run_optimizer():
    with tf.GradientTape() as g:
        pred = liner_reg(X)
        loss = mean_error(pred, Y)

    # Compute gradients
    gradients = g.gradient(loss, [W, b])

    # Update W and b following gradients
    optmizer.apply_gradients(zip(gradients, [W, b]))


# Run training for given number of steps

for step in range(1, epoches + 1):

    # Run the optimazation to update W and b values
    run_optimizer()

    if step % display_rate == 0:
        pred = liner_reg(X)
        loss = mean_error(pred, Y)

        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))

# Graphic display
plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, np.array(W * X + b), label='Fitted line')
plt.legend()
plt.show()
