import tensorflow as tf


# import sonnet as snt

class MLPGenNet(tf.keras.Model):
    def __init__(self,
                 name='mlp_generator'):
        super(MLPGenNet, self).__init__(name=name)
        self.dense_1 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense_2 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense_3 = tf.keras.layers.Dense(784, activation=tf.nn.leaky_relu)

    def call(self, inputs, batch_size=64, is_training=True):
        del is_training
        lay1_out = self.dense_1(inputs)
        lay2_out = self.dense_2(lay1_out)
        lay3_out = self.dense_3(lay2_out)
        out = tf.nn.tanh(lay3_out)
        return out


class MLPMesNet(tf.keras.Model):
    def __init__(self, num_outputs=2, name='mlp_measure'):
        super(MLPMesNet, self).__init__(name=name)
        self.dense_1 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense_2 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense_3 = tf.keras.layers.Dense(num_outputs, activation=tf.nn.leaky_relu)

    def call(self, inputs):
        lay1_out = self.dense_1(inputs)
        lay2_out = self.dense_2(lay1_out)
        out = self.dense_3(lay2_out)
        return out


class MLPGenNetSparse(tf.keras.Model):
    def __init__(self,
                 name='mlp_generator'):
        super(MLPGenNetSparse, self).__init__(name=name)
        self.dense_1 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense_2 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense_3 = tf.keras.layers.Dense(784, activation=tf.nn.leaky_relu)

    def call(self, inputs, batch_size=64, is_training=True):
        del is_training
        lay1_out = self.dense_1(inputs)
        lay2_out = self.dense_2(lay1_out)
        lay3_out = self.dense_3(lay2_out)
        out = tf.nn.tanh(lay3_out)
        return out


class MLPMesNetSparse(tf.keras.Model):
    def __init__(self, num_outputs=2, name='mlp_measure'):
        super(MLPMesNetSparse, self).__init__(name=name)
        self.dense_1 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense_2 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense_3 = tf.keras.layers.Dense(num_outputs, activation=tf.nn.leaky_relu)

    def call(self, inputs):
        lay1_out = self.dense_1(inputs)
        lay2_out = self.dense_2(lay1_out)
        out = self.dense_3(lay2_out)
        return out

