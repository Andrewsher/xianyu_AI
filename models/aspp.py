
from keras.regularizers import l2
from keras.layers import Reshape, Activation, Conv2D, MaxPooling2D, BatchNormalization, Input, DepthwiseConv2D, \
    add, Dropout, AveragePooling2D, Concatenate, Deconvolution2D, ReLU
from keras.models import Model
import keras.backend as K
from keras.engine import Layer, InputSpec
from keras.utils.conv_utils import normalize_tuple
from keras.backend.common import normalize_data_format


class BilinearUpsampling(Layer):
    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.upsampling = normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                   int(inputs.shape[2] * self.upsampling[1])))

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def conv2d_bn_relu(input, filters, kernel_size, strides=(1, 1), padding='same', dilation_rate=(1, 1),
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-5)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def depth_conv_bn_relu(input, filters, kernel_size, strides=(1, 1), padding='same', dilation_rate=(1, 1),
                   initializer='he_normal', regularizer=l2(1e-5)):
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, padding=padding,
                        depthwise_initializer=initializer, use_bias=False, depthwise_regularizer=regularizer)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding=padding,
               kernel_initializer=initializer, kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x



def aspp(x, input_shape, out_stride):

    b0 = conv2d_bn_relu(input=x, filters=256, kernel_size=(1, 1))

    b1 = conv2d_bn_relu(input=x, filters=256, kernel_size=(3, 3), dilation_rate=(2, 2))
    b2 = conv2d_bn_relu(input=x, filters=256, kernel_size=(3, 3), dilation_rate=(4, 4))
    b3 = conv2d_bn_relu(input=x, filters=256, kernel_size=(3, 3), dilation_rate=(6, 6))

    out_shape0 = int(input_shape[0] / out_stride)
    out_shape1 = int(input_shape[1] / out_stride)
    b4 = AveragePooling2D(pool_size=(out_shape0, out_shape1))(x)
    b4 = conv2d_bn_relu(input=b4, filters=256, kernel_size=(1, 1))
    b4 = BilinearUpsampling((out_shape0, out_shape1))(b4)

    x = Concatenate()([b4, b0, b1, b2, b3])
    return x

