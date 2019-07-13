from __future__ import print_function
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Lambda, Layer
from keras.layers import AveragePooling2D, Input, Flatten, MaxPool2D, Reshape, Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

from my_callback import my_TensorBoard
try:
    tf.gfile.DeleteRecursively('./logs')
except:
    print('Logs is not found!')


# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = False
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = False

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

class Attention_Layer(Layer):
    gamma = K.variable(0.0) # class variable

    def __init__(self, strides, num_filters, **kwargs):
        self.strides = strides
        self.num_filters = num_filters
        self.att_map = None
        self.att_feature = None
        super(Attention_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.trainable_weights = [Attention_Layer.gamma]

        super(Attention_Layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.attention(x, strides = self.strides, num_filters=self.num_filters)

    def compute_output_shape(self, input_shape):
        output_shape =(None, input_shape[1]//self.strides, input_shape[2]//self.strides, self.num_filters)
        return output_shape
    
    def get_att_map(self):
        return self.att_map
    
    def get_att_feature(self):
        return self.att_feature

    def attention(self, x,
                    num_filters=16,
                    kernel_size=3,
                    strides=1,
                    max_pooling=False):
        """2D Convolution-Batch Normalization-Activation for self-attention map
        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            h_feature (tensor): input tensor from feature map
            name (str): indicate name to value of attention easier
            num_filters (int): Conv2D number of filters output
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            max_pooling (bool): if True, apply max pooling 2D for inputs, f, g
        # Returns
            x (tensor): tensor as attention map
        """
        # get key, query and value
        f = resnet_layer(inputs=x,
                        num_filters=num_filters,
                        strides=strides)
        f = resnet_layer(inputs=f,
                        num_filters=num_filters,
                        activation=None,
                        batch_normalization=False)
        f = resnet_layer(inputs=f,
                        num_filters=num_filters,
                        kernel_size=1,
                        activation=None,
                        batch_normalization=False)  # linear layer [bs, h, w, c]

        g = resnet_layer(inputs=x,
                        num_filters=num_filters,
                        strides=strides)
        g = resnet_layer(inputs=g,
                        num_filters=num_filters,
                        activation=None,
                        batch_normalization=False)
        g = resnet_layer(inputs=g,
                        num_filters=num_filters,
                        kernel_size=1,
                        activation=None,
                        batch_normalization=False)  # linear layer [bs, h, w, c]

        h = resnet_layer(inputs=x,
                        num_filters=num_filters,
                        strides=strides)
        h = resnet_layer(inputs=h,
                        num_filters=num_filters,
                        activation=None,
                        batch_normalization=False)
        h = resnet_layer(inputs=h,
                        num_filters=num_filters,
                        kernel_size=1,
                        activation=None,
                        batch_normalization=False)  # linear layer [bs, h, w, c]

        # get output shape
        _, height, width, num_filters = K.int_shape(h)

        # flatten h and w
        f = Reshape((height*width, num_filters))(f)
        g = Reshape((height*width, num_filters))(g)
        h = Reshape((height*width, num_filters))(h)    
        
        # N = h * w
        s = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([g, f])  # [bs, N, N]

        att_map = Activation('softmax', name=att_name)(s) # attention map [0, 1] 
        
        att_feature = Lambda(lambda x: tf.matmul(x[0], x[1]))([att_map, h]) # residual attention map = att_map + 1.0
        att_feature = Lambda(lambda x: Attention_Layer.gamma*x[0] + x[1])([att_feature, h])
        att_feature = Reshape((height, width, num_filters))(att_feature)
        
        self.att_feature = att_feature
        self.att_map = att_map

        att_feature = resnet_layer(inputs=att_feature,
                                    num_filters=num_filters,
                                    kernel_size=1,
                                    activation=None,
                                    batch_normalization=False)  # linear layer [bs, h, w, c']
        
        return att_feature

def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape, name='img')
    x = resnet_layer(inputs=inputs,)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            # Self-attention
            att_name='layer_att'+str(stack)+str(res_block)
            y = Attention_Layer(strides, num_filters, name=att_name)(x)
            y = BatchNormalization()(y)

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                num_filters=num_filters,
                                kernel_size=1,
                                strides=strides,
                                activation=None,
                                batch_normalization=False) 
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

cb_tensorboard = my_TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, my_write='attention')

callbacks = [checkpoint, lr_reducer, lr_scheduler, cb_tensorboard]

# Run training, with or without data augmentation.
print('Not using data augmentation.')
model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True,
            callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
