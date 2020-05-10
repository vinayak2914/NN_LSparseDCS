import tensorflow as tf
import numpy as np

# import sonnet as snt

class SNConvGenNet(tf.keras.Model):
    def __init__(self,
                 name='snconv_generator'):
        super(SNConvGenNet, self).__init__(name=name)
        first_shape = [4, 4, 512]
        first_layer_nodes = np.prod(first_shape)
        self.dense_1 = tf.keras.layers.Dense(first_layer_nodes)

        self.conv2dT_1 = tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same')
        self.batch_norm1 = tf.keras.layers.BatchNormalization(scale=True,gamma_initializer='ones')
        self.actv_1 = tf.keras.layers.Activation(tf.nn.relu)

        self.conv2dT_2 = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')
        self.batch_norm2 = tf.keras.layers.BatchNormalization(scale=True,gamma_initializer='ones')
        self.actv_2 = tf.keras.layers.Activation(tf.nn.relu)

        self.conv2dT_3 = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')
        self.batch_norm3 = tf.keras.layers.BatchNormalization(scale=True,gamma_initializer='ones')
        self.actv_3 = tf.keras.layers.Activation(tf.nn.relu)

        self.conv2dT_4 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same')

    def call(self, inputs, batch_size=64, is_training=True):
        del is_training
        first_shape = [4, 4, 512]
        lay1_out = self.dense_1(inputs)
        first_tensor = tf.reshape(lay1_out, shape=[batch_size] + first_shape)

        conv1_out       = self.conv2dT_1(first_tensor)
        batch_nomr1_out = self.batch_norm1(conv1_out)
        actv1_out       = self.actv_1(batch_nomr1_out)

        conv2_out       = self.conv2dT_2(actv1_out)
        batch_nomr2_out = self.batch_norm2(conv2_out)
        actv2_out       = self.actv_2(batch_nomr2_out)

        conv3_out       = self.conv2dT_3(actv2_out)
        batch_nomr3_out = self.batch_norm3(conv3_out)
        actv3_out       = self.actv_3(batch_nomr3_out)

        conv4_out       = self.conv2dT_4(actv3_out)
        out = tf.nn.tanh(conv4_out)
        return out


class SNConvMesNet(tf.keras.Model):
    def __init__(self, num_outputs=2, name='snconv_measure'):
        super(SNConvMesNet, self).__init__(name=name)

        self.conv2d_1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.actv_1 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv2d_2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')
        self.actv_2 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv2d_3 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.actv_3 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv2d_4 = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')
        self.actv_4 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv2d_5 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.actv_5 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv2d_6 = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same')
        self.actv_6 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv2d_7 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')
        self.actv_7 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.flat = tf.keras.layers.Flatten()
        self.dense_out = tf.keras.layers.Dense(num_outputs)



    def call(self, inputs):
        conv1_out = self.conv2d_1(inputs)
        actv1_out = self.actv_1(conv1_out)

        conv2_out = self.conv2d_2(actv1_out)
        actv2_out = self.actv_2(conv2_out)

        conv3_out = self.conv2d_3(actv2_out)
        actv3_out = self.actv_3(conv3_out)

        conv4_out = self.conv2d_4(actv3_out)
        actv4_out = self.actv_4(conv4_out)

        conv5_out = self.conv2d_5(actv4_out)
        actv5_out = self.actv_5(conv5_out)

        conv6_out = self.conv2d_6(actv5_out)
        actv6_out = self.actv_6(conv6_out)

        conv7_out = self.conv2d_7(actv6_out)
        actv7_out = self.actv_7(conv7_out)

        flat_out = self.flat(actv7_out)
        out = self.dense_out(flat_out)

        return out


class SNConvGenNetSparse(tf.keras.Model):
    def __init__(self,
                 name='snconv_generatorsparse'):
        super(SNConvGenNetSparse, self).__init__(name=name)
        first_shape = [4, 4, 512]
        first_layer_nodes = np.prod(first_shape)
        self.dense_1 = tf.keras.layers.Dense(first_layer_nodes)

        self.conv2dT_1 = tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same')
        self.batch_norm1 = tf.keras.layers.BatchNormalization(scale=True,gamma_initializer='ones')
        self.actv_1 = tf.keras.layers.Activation(tf.nn.relu)

        self.conv2dT_2 = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')
        self.batch_norm2 = tf.keras.layers.BatchNormalization(scale=True,gamma_initializer='ones')
        self.actv_2 = tf.keras.layers.Activation(tf.nn.relu)

        self.conv2dT_3 = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')
        self.batch_norm3 = tf.keras.layers.BatchNormalization(scale=True,gamma_initializer='ones')
        self.actv_3 = tf.keras.layers.Activation(tf.nn.relu)

        self.conv2dT_4 = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same')

    def call(self, inputs, batch_size=64, is_training=True):
        del is_training
        first_shape = [4, 4, 512]
        lay1_out = self.dense_1(inputs)
        first_tensor = tf.reshape(lay1_out, shape=[batch_size] + first_shape)

        conv1_out       = self.conv2dT_1(first_tensor)
        batch_nomr1_out = self.batch_norm1(conv1_out)
        actv1_out       = self.actv_1(batch_nomr1_out)

        conv2_out       = self.conv2dT_2(actv1_out)
        batch_nomr2_out = self.batch_norm2(conv2_out)
        actv2_out       = self.actv_2(batch_nomr2_out)

        conv3_out       = self.conv2dT_3(actv2_out)
        batch_nomr3_out = self.batch_norm3(conv3_out)
        actv3_out       = self.actv_3(batch_nomr3_out)

        conv4_out       = self.conv2dT_4(actv3_out)
        out = tf.nn.tanh(conv4_out)
        return out


class SNConvMesNetSparse(tf.keras.Model):
    def __init__(self, num_outputs=2, name='snconv_measuresparse'):
        super(SNConvMesNetSparse, self).__init__(name=name)

        self.conv2d_1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.actv_1 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv2d_2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')
        self.actv_2 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv2d_3 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.actv_3 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv2d_4 = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')
        self.actv_4 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv2d_5 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.actv_5 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv2d_6 = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same')
        self.actv_6 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.conv2d_7 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')
        self.actv_7 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.flat = tf.keras.layers.Flatten()
        self.dense_out = tf.keras.layers.Dense(num_outputs)



    def call(self, inputs):
        conv1_out = self.conv2d_1(inputs)
        actv1_out = self.actv_1(conv1_out)

        conv2_out = self.conv2d_2(actv1_out)
        actv2_out = self.actv_2(conv2_out)

        conv3_out = self.conv2d_3(actv2_out)
        actv3_out = self.actv_3(conv3_out)

        conv4_out = self.conv2d_4(actv3_out)
        actv4_out = self.actv_4(conv4_out)

        conv5_out = self.conv2d_5(actv4_out)
        actv5_out = self.actv_5(conv5_out)

        conv6_out = self.conv2d_6(actv5_out)
        actv6_out = self.actv_6(conv6_out)

        conv7_out = self.conv2d_7(actv6_out)
        actv7_out = self.actv_7(conv7_out)

        flat_out = self.flat(actv7_out)
        out = self.dense_out(flat_out)

        return out

