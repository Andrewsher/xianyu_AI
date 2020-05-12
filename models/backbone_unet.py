from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing, get_backbone, get_feature_layers
from keras.layers import BatchNormalization, UpSampling2D, Conv2D, LeakyReLU, ReLU, Concatenate, Dropout, Input, \
    AveragePooling2D, Activation, AtrousConv2D
from keras.regularizers import l2
from keras.models import Model

# from .aspp import aspp
# from .non_local import block_n
# from keras.applications import ResNet50


def conv2d_bn_leakyrelu(input, filters, kernel_size, strides=(1,1), padding='same', dilation_rate=(1,1),
                   kernel_initializer='he_normal', kernel_regularizer=l2(1e-5), name=None):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name=name)(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def create_backbone_unet(input_shape=(256, 256, 3), pretrained_weights_file=None, backbone='vgg16'):
    # build model for segmenting all masks
    if pretrained_weights_file == 'imagenet':
        encoder_weights = 'imagenet'
    else:
        encoder_weights = None
    if pretrained_weights_file != None:
        encoder_freeze = True
    else:
        encoder_freeze = False
    # input
    # input = Input(shape=input_shape, name='input_layer_0')
    # x = conv2d_bn_leakyrelu(input, filters=16, kernel_size=3, dilation_rate=(2, 2), name='dilation_1')
    # x = conv2d_bn_leakyrelu(x, filters=16, kernel_size=3, dilation_rate=(2, 2), name='dilation_2')
    # x = ReLU(name='input_1')(x)
    # x = Input(shape=input_shape, tensor=x)
    # print(x)
    # encoder
    encoder = get_backbone(name=backbone, input_shape=input_shape, weights=encoder_weights, include_top=False)
    # get skip connections
    stages = []
    feature_layers = get_feature_layers(name=backbone, n=5)
    for feature_layer in feature_layers:
        stages.append(encoder.get_layer(feature_layer).output)

    # decoder
    x = encoder.output
    # x = block_n(x)
    # x = aspp(x, input_shape=input_shape, out_stride=32)
    # x = conv2d_bn_leakyrelu(x, filters=512, kernel_size=1)
    x = Dropout(0.2)(x)

    for i in range(5):
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        if i < len(stages):
            x = conv2d_bn_leakyrelu(x, filters=2 ** (8 - i), kernel_size=3)
            x = Concatenate()([x, stages[i]])
        x = conv2d_bn_leakyrelu(x, filters=2 ** (8 - i), kernel_size=3)
        x = conv2d_bn_leakyrelu(x, filters=2 ** (8 - i), kernel_size=3)

    # output
    x = Conv2D(filters=4, kernel_size=1)(x)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Activation('softmax')(x)

    # freeze encoder
    if encoder_freeze:
        for layer in encoder.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

    # create model
    model = Model(input=encoder.input, output=x)
    print('Create U-Net, input shape = {}, output shape = {}'.format(input_shape, model.output.shape))

    if pretrained_weights_file != None and pretrained_weights_file != 'imagenet':
        model.load_weights(pretrained_weights_file, skip_mismatch=True, by_name=True)
        print('Weights loaded from', pretrained_weights_file)

    return model


if __name__ == '__main__':
    model = create_backbone_unet(backbone='resnet18')
    model.summary()
