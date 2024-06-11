import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import numpy as np

epsilon = 1e-7
target_size = (120, 120)
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5
alpha = 0.0005
epochs = 50


def safe_norm(v, axis=-1, epsilon=1e-7):
    v_ = tf.reduce_sum(tf.square(v), axis = axis, keepdims=True)
    return tf.sqrt(v_ + epsilon)

def loss_function(v, reconstructed_image, y, y_image, **params):
    prediction = safe_norm(v)
    prediction = tf.reshape(prediction, [-1, params["no_of_secondary_capsules"]])

    left_margin = tf.square(tf.maximum(0.0, m_plus - prediction))
    right_margin = tf.square(tf.maximum(0.0, prediction - m_minus))

    l = tf.add(y * left_margin, lambda_ * (1.0 - y) * right_margin)

    margin_loss = tf.reduce_mean(tf.reduce_sum(l, axis=-1))

    y_image_flat = Flatten()(y_image)
    reconstruction_loss = tf.reduce_mean(tf.square(y_image_flat - reconstructed_image))

    loss = tf.add(margin_loss, alpha * reconstruction_loss)

    return loss

class CapsuleNetwork(tf.keras.Model):
    def __init__(self, no_of_conv_kernels, no_of_primary_capsules, primary_capsule_vector, no_of_secondary_capsules, secondary_capsule_vector, r):
        super(CapsuleNetwork, self).__init__()
        self.no_of_conv_kernels = no_of_conv_kernels
        self.no_of_primary_capsules = no_of_primary_capsules
        self.primary_capsule_vector = primary_capsule_vector
        self.no_of_secondary_capsules = no_of_secondary_capsules
        self.secondary_capsule_vector = secondary_capsule_vector
        self.r = r

        with tf.name_scope("Variables") as scope:
            self.convolution = Conv2D(self.no_of_conv_kernels, [9,9], strides=[2,2], name='ConvolutionLayer', activation='relu')
            # self.convolution = Sequential([
                # Conv2D(self.no_of_conv_kernels, [9,9], strides=[2,2], name='ConvolutionLayer-1', activation='relu'),
                # Conv2D(self.no_of_conv_kernels, [9,9], strides=[2,2], name='ConvolutionLayer-2', activation='relu'), # 8x8x32 = 9216
                # ])
            self.primary_capsule = Conv2D(self.no_of_primary_capsules * self.primary_capsule_vector,
                                                          [9,9], strides=[2,2], name="PrimaryCapsule")
            self.w = tf.Variable(tf.random_normal_initializer()(shape=[1, 9216, self.no_of_secondary_capsules, self.secondary_capsule_vector, self.primary_capsule_vector]), dtype=tf.float32, name="PoseEstimation", trainable=True)
            self.dense_1 = Dense(units = 512, activation='relu')
            self.dense_2 = Dense(units = 1024, activation='relu')
            self.dense_3 = Dense(units = np.product(target_size)*3, activation='sigmoid', dtype='float32')

    def build(self, input_shape):
        pass

    def squash(self, s):
        with tf.name_scope("SquashFunction") as scope:
            s_norm = tf.norm(s, axis=-1, keepdims=True)
            return tf.square(s_norm)/(1 + tf.square(s_norm)) * s/(s_norm + epsilon)

    @tf.function
    def call(self, inputs):
        input_x, y = inputs
        v = self.predict_capsule_output(input_x)

        with tf.name_scope("Masking") as scope:
            y = tf.expand_dims(y, axis=-1) # y.shape: (None, 2, 1)
            y = tf.expand_dims(y, axis=1) # y.shape: (None, 1, 2, 1)
            mask = tf.cast(y, dtype=tf.float32) # mask.shape: (None, 1, 2, 1)
            v_masked = tf.multiply(mask, v) # v_masked.shape: (None, 1, 2, 16)

        with tf.name_scope("Reconstruction") as scope:
            # v_ = tf.reshape(v_masked, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) # v_.shape: (None, 32)
            v_ = Flatten()(v_masked)
            reconstructed_image = self.dense_1(v_) # reconstructed_image.shape: (None, 512)
            reconstructed_image = self.dense_2(reconstructed_image) # reconstructed_image.shape: (None, 1024)
            reconstructed_image = self.dense_3(reconstructed_image) # reconstructed_image.shape: (None, np.product(target_size)*3)

        return v, reconstructed_image

    @tf.function
    def predict_capsule_output(self, inputs):
        x = self.convolution(inputs) # x.shape: (None, 244, 244, 256)
        x = self.primary_capsule(x) # x.shape: (None, 7, 7, 256)
        # 24 x 24 x 16 = 9216
        flatten_size = self.no_of_primary_capsules * x.shape[1] * x.shape[2]

        with tf.name_scope("CapsuleFormation") as scope:
            u = tf.reshape(x, (-1, flatten_size, 8)) # u.shape: (None, 9216, 8)
            u = tf.expand_dims(u, axis=-2) # u.shape: (None, 9216, 1, 8)
            u = tf.expand_dims(u, axis=-1) # u.shape: (None, 9216, 1, 8, 1)
            u_hat = tf.matmul(self.w, u) # u_hat.shape: (None, 9216, 2, 16, 1)
            u_hat = tf.squeeze(u_hat, [4]) # u_hat.shape: (None, 9216, 2, 16)

        with tf.name_scope("DynamicRouting") as scope:
            b = tf.zeros((inputs.shape[0], flatten_size, self.no_of_secondary_capsules, 1)) # b.shape: (None, 9216, 2, 1)
            for i in range(self.r): # self.r = 3
                c = tf.nn.softmax(b, axis=-2) # c.shape: (None, 9216, 2, 1)
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True) # s.shape: (None, 1, 2, 16)
                v = self.squash(s) # v.shape: (None, 1, 2, 16)
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4]) # agreement.shape: (None, 1152, 10, 1)
                # Before matmul following intermediate shapes are present, they are not assigned to a variable but just for understanding the code.
                # u_hat.shape (Intermediate shape) : (None, 9216, 2, 16, 1)
                # v.shape (Intermediate shape): (None, 1, 2, 16, 1)
                # Since the first parameter of matmul is to be transposed its shape becomes:(None, 9216, 2, 1, 16)
                # Now matmul is performed in the last two dimensions, and others are broadcasted
                # Before squeezing we have an intermediate shape of (None, 9216, 2, 1, 1)
                b += agreement
        return v

    @tf.function
    def regenerate_image(self, inputs):
        with tf.name_scope("Reconstruction") as scope:
            v_ = tf.reshape(inputs, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) # v_.shape: (None, 32)
            reconstructed_image = self.dense_1(v_) # reconstructed_image.shape: (None, 512)
            reconstructed_image = self.dense_2(reconstructed_image) # reconstructed_image.shape: (None, 1024)
            reconstructed_image = self.dense_3(reconstructed_image) # reconstructed_image.shape: (None, np.product(target_size)*3)
        return reconstructed_image