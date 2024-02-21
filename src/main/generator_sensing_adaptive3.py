import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
import numpy as np
from main import model_util
from main.config import Config
from main.smpl import Smpl

def real_awgn(x, stddev):
    """Implements the real additive white gaussian noise channel.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # additive white gaussian noise
    awgn = tf.random.normal(tf.shape(x), 0, stddev, dtype=tf.float32)
    y = x + awgn

    return y


def fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    if h is None:
        h = tf.complex(
            tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),
            tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),
        )

    # additive white gaussian noise
    awgn = tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )

    return (h * x + stddev * awgn), h


def phase_invariant_fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise. Also assumes that phase shift
    introduced by the fading channel is known at the receiver, making
    the model equivalent to a real slow fading channel.

    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    if h is None:
        n1 = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2), dtype=tf.float32)
        n2 = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2), dtype=tf.float32)

        h = tf.sqrt(tf.square(n1) + tf.square(n2))

    # additive white gaussian noise
    awgn = tf.random.normal(tf.shape(x), 0, stddev / np.sqrt(2), dtype=tf.float32)

    return (h * x + awgn), h

class Channel(tf.keras.Model):
    def __init__(self, channel_type, channel_snr, name="channel", **kwargs):
        super(Channel, self).__init__(name=name, **kwargs)
        self.channel_type = channel_type
        self.channel_snr = channel_snr

    def call(self, inputs):
        (encoded_img, prev_h) = inputs
        inter_shape = tf.shape(encoded_img)
        # reshape array to [-1, dim_z]
        z = layers.Flatten()(encoded_img)
        # convert from snr to std
        print("channel_snr: {}".format(self.channel_snr))
        noise_stddev = np.sqrt(10 ** (-self.channel_snr / 10))

        # Add channel noise
        if self.channel_type == "awgn":
            dim_z = tf.shape(z)[1]
            # normalize latent vector so that the average power is 1
            z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
                z, axis=1
            )
            z_out = real_awgn(z_in, noise_stddev)
            h = tf.ones_like(z_in)  # h just makes sense on fading channels

        elif self.channel_type == "fading":
            dim_z = tf.shape(z)[1] // 2
            # convert z to complex representation
            z_in = tf.complex(z[:, :dim_z], z[:, dim_z:])
            # normalize the latent vector so that the average power is 1
            z_norm = tf.reduce_sum(
                tf.math.real(z_in * tf.math.conj(z_in)), axis=1, keepdims=True
            )
            z_in = z_in * tf.complex(
                tf.sqrt(tf.cast(dim_z, dtype=tf.float32) / z_norm), 0.0
            )
            z_out, h = fading(z_in, noise_stddev, prev_h)
            # convert back to real
            z_out = tf.concat([tf.math.real(z_out), tf.math.imag(z_out)], 1)

        elif self.channel_type == "fading-real":
            # half of the channels are I component and half Q
            dim_z = tf.shape(z)[1] // 2
            # normalization
            z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
                z, axis=1
            )
            z_out, h = phase_invariant_fading(z_in, noise_stddev, prev_h)

        else:
            raise Exception("This option shouldn't be an option!")

        # convert signal back to intermediate shape
        z_out = tf.reshape(z_out, inter_shape)
        # compute average power
        avg_power = tf.reduce_mean(tf.math.real(z_in * tf.math.conj(z_in)))
        # add avg_power as layer's metric
        return z_out, avg_power, h

class Regressor(tf.keras.Model):

    def __init__(self):
        super(Regressor, self).__init__(name='regressor')
        self.config = Config()

        self.mean_theta = tf.Variable(model_util.load_mean_theta(), name='mean_theta', trainable=True)

        self.fc_one = layers.Dense(1024, name='fc_0')
        self.dropout_one = layers.Dropout(0.5)
        self.fc_two = layers.Dense(1024, name='fc_1')
        self.dropout_two = layers.Dropout(0.5)
        variance_scaling = tf.keras.initializers.VarianceScaling(.01, mode='fan_avg', distribution='uniform')
        self.fc_out = layers.Dense(85, kernel_initializer=variance_scaling, name='fc_out')

    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, 2048)
        assert inputs.shape[1:] == shape[1:], 'shape mismatch: should be {} but is {}'.format(shape, inputs.shape)

        batch_theta = tf.tile(self.mean_theta, [batch_size, 1])
        thetas = tf.TensorArray(tf.float32, self.config.ITERATIONS)
        for i in range(self.config.ITERATIONS):
            # [batch x 2133] <- [batch x 2048] + [batch x 85]
            total_inputs = tf.concat([inputs, batch_theta], axis=1)
            batch_theta = batch_theta + self._fc_blocks(total_inputs, **kwargs)
            thetas = thetas.write(i, batch_theta)

        return thetas.stack()

    def _fc_blocks(self, inputs, **kwargs):
        x = self.fc_one(inputs, **kwargs)
        x = tf.nn.relu(x)
        x = self.dropout_one(x, **kwargs)
        x = self.fc_two(x, **kwargs)
        x = tf.nn.relu(x)
        x = self.dropout_two(x, **kwargs)
        x = self.fc_out(x, **kwargs)
        return x
@tf.custom_gradient
def Quantizer(x):
    L=16.   # 16
    y=tf.zeros_like(x)
    for l in range(16):
        l=tf.cast(l,tf.float32)
        y=tf.where(tf.math.logical_and(x>=tf.cast(l/(L-1)-0.5/(L-1),tf.float32),x<tf.cast(l/(L-1)+0.5/(L-1),tf.float32)),l*tf.ones_like(x),y)
    def grad(upstream):
        multiplier= (L-1)*tf.ones_like(x)
        return upstream*multiplier
    # y=tf.where(x>0.5,tf.ones_like(x),tf.zeros_like(x))
    return y, grad

@tf.custom_gradient
def Mask(x):
    n=200
    L=15.
    m=tf.zeros_like(x)
    m=tf.tile(m,[1, 1, 1, n])

    batch = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    idx_value = tf.range(n)
    idx_value = tf.expand_dims(tf.expand_dims(tf.expand_dims(idx_value, axis=0),axis=0),axis=0)
    idx_value = tf.tile(idx_value, [batch, height, width, 1])
    idx_value = tf.cast(idx_value,tf.float32)

    m=tf.where(idx_value<(n/L)*x, tf.ones_like(m), tf.zeros_like(m))
    def grad(upstream):
        multiplier = tf.where(tf.math.logical_and((tf.math.ceil(idx_value*L/n)>=x-1),(tf.math.ceil(idx_value*L/n)< x+2)), (1/3.)*tf.ones_like(idx_value), tf.zeros_like(idx_value))
        return tf.reduce_sum(multiplier*upstream,keepdims=True)
    return m, grad

class Sampling_plus_init_reconstruction_network(tf.keras.Model):

    def __init__(self):
        super(Sampling_plus_init_reconstruction_network, self).__init__(name='sampling')



        # self.init_sampling_weight = np.random.randn(1, 1, 1, 32*32*3, 103)
        # self.sampling_weight = tf.Variable( self.init_sampling_weight, dtype=tf.float32, name='sampling_weight', trainable=True)
        self.sensing_importance = layers.Conv2D(20, 32, strides=[32, 32], use_bias=False, padding='valid',
                                                activation=None)
        self.conv1 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')

        self.conv4 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.conv5 = layers.Conv2D(1, 1, padding='same', activation='sigmoid')

        # self.conv1 = layers.Conv2D(256, 3, padding='same', activation='relu')
        # self.conv2 = layers.Conv2D(256, 3, padding='same', activation='relu')
        # self.conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')
        # self.conv4 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.conv6 = layers.Conv2D(128, 1, padding='same', activation='relu')
        self.conv7 = layers.Conv2D(128, 1, padding='same', activation='relu')
        self.conv8 = layers.Conv2D(128, 1, padding='same', activation='relu')
        self.conv9 = layers.Conv2D(220 * 220, 1, padding='same', activation=None)

        self.sensing_conv = layers.Conv2D(200, 32, strides=[32, 32], use_bias=False, padding='valid', activation=None)
        # self.conv4 = layers.Conv2D(512, 1, padding='same', activation='relu')
        # self.conv5 = layers.Conv2D(512, 1, padding='same', activation='relu')
        # self.conv6 = layers.Conv2D(512, 1, padding='same', activation='relu')
        self.re_conv = layers.Conv2D(32 * 32 * 3, 1, strides=[1, 1], use_bias=False, padding='valid', activation=None)
    def call(self, inputs, **kwargs):
        first_x = self.sensing_importance(inputs)
        first_x_feature = self.conv1(tf.stop_gradient(first_x))
        first_x_feature = self.conv2(first_x_feature)
        first_x_feature = self.conv3(first_x_feature)
        first_x_feature = self.conv4(first_x_feature)
        rate = self.conv5(first_x_feature)
        weights = self.conv6(tf.stop_gradient(first_x_feature))
        weights = self.conv7(weights)
        weights = self.conv8(weights)
        weights = self.conv9(weights)
        # rat_n=rate.numpy()



        bit = Quantizer(rate)
        m = Mask(bit)

        m_mean, m_var = tf.nn.moments(tf.reduce_mean(tf.reduce_sum(m, axis=-1), axis=[1, 2]),axes=0)
        # m_mean_n=m_mean.numpy()

        x = self.sensing_conv(inputs)

        # m_tiled=tf.tile(m,[batch,1,1,1])
        x = x * m

        concated_sensing_o=tf.concat([first_x, x], axis=-1)
        batch = tf.shape(concated_sensing_o)[0]
        height = tf.shape(concated_sensing_o)[1]
        width = tf.shape(concated_sensing_o)[2]
        channels = tf.shape(concated_sensing_o)[3]
        concated_sensing=tf.reshape(concated_sensing_o,[batch,height,width,1,channels])
        weights_weight=tf.reshape(weights,[batch,height,width,220,220])
        # weights_biases = tf.reshape(weights[:, :, :, -450:], [batch, height, width, 450])
        concated_sensing2=tf.squeeze(tf.matmul(concated_sensing,weights_weight),axis=3)
        concated_sensing=concated_sensing_o+concated_sensing2


        # x = self.conv4(tf.concat([first_x, x, tf.stop_gradient(rate)], axis=-1))
        # x = self.conv5(x)
        # x = self.conv6(x)
        x = self.re_conv(concated_sensing)
        # x = tf.reshape(x, [-1, 7, 7, 32, 32, 3])
        # x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        # #
        # x = tf.reshape(x, [-1, 224, 224, 3])

        return x, rate, m_mean, m_var


class Deep_reconstruction_network(tf.keras.Model):

    def __init__(self):
        super(Deep_reconstruction_network, self).__init__(name='deep_re')
        self.in_conv1 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.res1_1 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.res1_2 = layers.Conv2D(256, 3, padding='same', activation=None)
        self.out_conv1 = layers.Conv2D(2 * 32 * 3, 3, padding='same', activation='relu')

        self.in_conv2 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.res2_1 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.res2_2 = layers.Conv2D(128, 3, padding='same', activation=None)
        self.out_conv2 = layers.Conv2D(2 * 8 * 3, 3, padding='same', activation='relu')

        self.in_conv3 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.res3_1 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.res3_2 = layers.Conv2D(64, 3, padding='same', activation=None)
        self.out_conv3 = layers.Conv2D(3, 3, padding='same', activation=None)


    def call(self, inputs, **kwargs):
        x = tf.nn.depth_to_space(inputs, 4)

        x = self.in_conv1(x)
        x_1 = self.res1_1(x)
        x_1 = self.res1_2(x_1)
        x = x + x_1
        x = self.out_conv1(x)

        x = tf.nn.depth_to_space(x, 2)
        x = self.in_conv2(x)
        x_1 = self.res2_1(x)
        x_1 = self.res2_2(x_1)
        x = x + x_1
        x = self.out_conv2(x)

        x = tf.nn.depth_to_space(x, 4)
        x = self.in_conv3(x)
        x_1 = self.res3_1(x)
        x_1 = self.res3_2(x_1)
        x = x + x_1
        x = self.out_conv3(x)

        return x



class Generator(tf.keras.Model):

    def __init__(self,name='generator'):
        super(Generator, self).__init__(name=name)
        self.config = Config()

        self.enc_shape = self.config.ENCODER_INPUT_SHAPE
        self.resnet50V2 = ResNet50V2(include_top=False, weights= None, input_shape=self.enc_shape, pooling='avg')
        self._set_resnet_arg_scope()
        self.resnet50V2.summary()
        # self.channel = Channel("awgn", 1, name="channel_output")

        # self.encoder=Encoder()
        # self.encoder.summary()
        self.regressor = Regressor()
        self.smpl = Smpl()
        self.sensing=Sampling_plus_init_reconstruction_network()
        self.reconstruct=Deep_reconstruction_network()

    def _set_resnet_arg_scope(self):
        """This method acts similar to TF 1.x contrib's slim `resnet_arg_scope()`.
            It overrides
        """
        vs_initializer = tf.keras.initializers.VarianceScaling(2.0)
        l2_regularizer = tf.keras.regularizers.l2(self.config.GENERATOR_WEIGHT_DECAY)
        for layer in self.resnet50V2.layers:
            if isinstance(layer, layers.Conv2D):
                # original implementations slim `resnet_arg_scope` additionally sets
                # `normalizer_fn` and `normalizer_params` which in TF 2.0 need to be implemented
                # as own layers. This is not possible using keras ResNet50V2 application.
                # Nevertheless this is not needed as training seems to be likely stable.
                # See https://www.tensorflow.org/guide/migrate#a_note_on_slim_contriblayers for more
                # migration insights
                # setattr(layer, 'padding', 'same')
                setattr(layer, 'kernel_initializer', vs_initializer)
                setattr(layer, 'kernel_regularizer', l2_regularizer)
            if isinstance(layer, layers.BatchNormalization):
                setattr(layer, 'momentum', 0.997)
                setattr(layer, 'epsilon', 1e-5)
            # if isinstance(layer, layers.MaxPooling2D):
            #     setattr(layer, 'padding', 'same')

    def call(self, inputs, **kwargs):
        check = inputs.shape[1:] == self.enc_shape
        assert check, 'shape mismatch: should be {} but is {}'.format(self.enc_shape, inputs.shape)

       #  features = self.resnet50V2(inputs, **kwargs)
       #  # self.encoder.summary()
       # # prev_chn_gain = None
       # # features, avg_power, chn_gain = self.channel((features, prev_chn_gain))
       #  thetas = self.regressor(features, **kwargs)
       #
       #  outputs = []
       #  for i in range(self.config.ITERATIONS):
       #      theta = thetas[i, :]
       #      outputs.append(self._compute_output(theta, **kwargs))
        features, rate, m_mean, m_var=self.sensing(inputs)
        # outputs=features
        outputs=self.reconstruct(features)

        return outputs, tf.reduce_mean(rate), m_mean, m_var

    def _compute_output(self, theta, **kwargs):
        cams = theta[:, :self.config.NUM_CAMERA_PARAMS]
        pose_and_shape = theta[:, self.config.NUM_CAMERA_PARAMS:]
        vertices, joints_3d, rotations = self.smpl(pose_and_shape, **kwargs)
        joints_2d = model_util.batch_orthographic_projection(joints_3d, cams)
        shapes = theta[:, -self.config.NUM_SHAPE_PARAMS:]

        return tf.tuple([vertices, joints_2d, joints_3d, rotations, shapes, cams])
