###################
###             ###
##  BUILD MODEL  ##
###             ###
###################

from tensorflow.keras.layers import Dense, Conv1D, Conv1DTranspose, Flatten, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import Model


def conv_block(
    x,
    filters,
    activation,
    kernel_size,
    strides,
    padding="same",
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=0.4,
):
    x = Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)

    if use_bn:
        x = BatchNormalization()(x)

    if activation:
      x = activation(x)

    if use_dropout:
        x = Dropout(drop_value)(x)
    return x


def upsample_block(
    x,
    filters,
    activation,
    kernel_size,
    strides,
    padding="same",
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    drop_value=0.3,
):
    x = Conv1DTranspose (filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)

    if use_bn:
        x = BatchNormalization()(x)

    if activation:
        x = activation(x)

    if use_dropout:
        x = Dropout(drop_value)(x)

    return x

class ConvLayer(Model):
    def __init__(self, out_filters, stride, k_size):
      super(ConvLayer, self).__init__()
      self.conv = Conv1D(filters=out_filters, kernel_size=k_size, strides=stride, padding = 'same')
      self.bn = BatchNormalization()
      self.lr = LeakyReLU()
      self.dropout = Dropout(0.2)
    
    def call(self, x, training):
      if training:
        x = self.dropout(x)
      x = self.conv(x)
      x = self.lr(x)
      x = self.bn(x)
      return x
