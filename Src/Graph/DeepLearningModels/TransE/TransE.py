
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def main():
    print("Welcome to TransE")

    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    print(opt)
    var = tf.Variable(1.0)
    print(var.numpy())
    val0 = var.value()
    print(val0)
    loss = lambda: (var ** 2) / 2.0  # d(loss)/d(var1) = var1
    print(loss)
    # First step is `-learning_rate*grad`
    step_count = opt.minimize(loss, [var]).numpy()
    print(step_count)
    val1 = var.value()
    print(val1)
    diff = (val0 - val1).numpy()
    print(diff)
    # On later steps, step-size increases because of momentum
    step_count = opt.minimize(loss, [var]).numpy()
    print(step_count)
    val2 = var.value()
    print(val2)
    diff2 = (val1 - val2).numpy()
    print(diff2)


    x_vals = np.random.normal(1, 0.1, 100)
    y_vals = np.repeat(10., 100)
    batch_size = 20
    X_data = tf.Variable(shape=[None, 1], dtype=tf.float32)
    Y_target = tf.Variable(shape=[None, 1], dtype=tf.float32)
    W = tf.Variable(tf.constant(0., shape=[1, 1]))
    Y_pred = tf.matmul(X_data, W)
    loss = tf.reduce_mean(tf.square(Y_pred - Y_target))
    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    step_count = opt.minimize(loss,[W])
    loss_batch = []
    for i in range(100):
        # pick a random 20 data points
        rand_index = np.random.choice(100, size=batch_size)
        x_batch = np.transpose([x_vals[rand_index]])  # Transpose to the correct shape
        y_batch = np.transpose([y_vals[rand_index]])
        sess.run(train_step, feed_dict={X_data: x_batch, Y_target: y_batch})
        # Print the result after 5 intervals
        if (i + 1) % 5 == 0:
            print('Step #', str(i + 1), 'W = ', str(sess.run(W)))
            temp_loss = sess.run(loss, feed_dict={X_data: x_batch, Y_target: y_batch})
            loss_batch.append(temp_loss)
            print('Loss = ', temp_loss)



if __name__ == "__main__":
    main()