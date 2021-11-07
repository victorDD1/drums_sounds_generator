##################
###            ###
##  GAN MODELS  ##
###            ###
##################

### Define GAN models

from os import read
import tensorflow as tf
import soundfile as sf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Conv1D, Flatten, BatchNormalization, LeakyReLU, Dropout, Reshape, Input
from tensorflow.keras import Model, Sequential
from utils.bulid_model import upsample_block, conv_block

###
### GENERATOR
###

def get_generator(latent_dim, n_layers):

  # Parameters
  noise = Input(shape=(latent_dim,))
  in_filter_shape = latent_dim
  dense_shape = 2048

  filter_list = [in_filter_shape//2**i for i in range(n_layers)]
  stride_list = [2]*(n_layers//2 + n_layers%2 + 1) + [4]*(n_layers//2  - 1)
  k_size_list = [3]*(n_layers//2 + n_layers%2) + [5]*(n_layers//2)

  # Layers
  x = Dense(2048, use_bias=False)(noise)
  x = BatchNormalization()(x)
  x = LeakyReLU(0.2)(x)

  x = Reshape((dense_shape//in_filter_shape, in_filter_shape))(x)

  for i in range(n_layers):
    x = upsample_block(
      x,
      filter_list[i],
      LeakyReLU(0.1),
      k_size_list[i],
      stride_list[i],
      use_bias=False,
      use_bn=True,
      padding='same',
      use_dropout=False,
    )
  
  x = upsample_block(x, 1, layers.Activation('tanh'), kernel_size=5, strides=1, use_bias=True, use_bn=True)

  generator_model = Model(noise, x, name='generator')
  return generator_model


###
### DISCRIMINATOR
###

def get_discriminator(sound_shape, n_layers):

  # Parameters
  filter_list = [2**(i) for i in range(n_layers)]
  stride_list = [4]*(n_layers//2 + n_layers%2) + [2]*(n_layers//2)
  k_size_list = [8]*(n_layers//2 + n_layers%2) + [3]*(n_layers//2)

  # Layers
  sound_input = Input(shape=sound_shape)
  x = sound_input

  for i in range(n_layers):
    x = conv_block(
      x,
      filter_list[i],
      LeakyReLU(0.1),
      k_size_list[i],
      stride_list[i],
      use_bias=True,
      use_bn=False,
      padding='same',
      use_dropout= bool(i>=n_layers//2),
      drop_value=0.3
    )
  
  x = layers.Flatten()(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Dense(1)(x)

  discriminator_model = Model(sound_input, x, name='discriminator')
  return discriminator_model

###
### GAN
###

class GAN(Model):
  '''
  https://keras.io/examples/generative/wgan_gp/#create-the-wgangp-model
  '''
  def __init__(
    self,
    discriminator,
    generator,
    latent_dim,
    discriminator_extra_steps,
    gp_weight=10.0,
  ):
    super(GAN, self).__init__()
    self.discriminator = discriminator
    self.generator = generator
    self.latent_dim = latent_dim
    self.d_steps = discriminator_extra_steps
    self.gp_weight = gp_weight

  def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
      super(GAN, self).compile()
      self.d_optimizer = d_optimizer
      self.g_optimizer = g_optimizer
      self.d_loss_fn = d_loss_fn
      self.g_loss_fn = g_loss_fn

  def gradient_penalty(self, batch_size, real_sounds, fake_sounds):
    """ Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    # Get the interpolated image
    alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
    diff = fake_sounds - real_sounds
    interpolated = real_sounds + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = self.discriminator(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

  def train_step(self, real_samples):
    # Get the batch size
    batch_size = tf.shape(real_samples)[0]
    for i in range(self.d_steps):
        # Get the latent vector
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )
        with tf.GradientTape() as tape:
            # Generate fake sounds from the latent vector
            fake_sounds = self.generator(random_latent_vectors, training=True)
            # Get the logits for the fake sounds
            fake_logits = self.discriminator(fake_sounds, training=True)
            # Get the logits for the real sounds
            real_logits = self.discriminator(real_samples, training=True)

            # Calculate the discriminator loss using the fake and real image logits
            d_cost = self.d_loss_fn(real_logits, fake_logits)
            # Calculate the gradient penalty  
            gp = self.gradient_penalty(batch_size, real_samples, fake_sounds)
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp * self.gp_weight

        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.d_optimizer.apply_gradients(
            zip(d_gradient, self.discriminator.trainable_variables)
        )

    # Train the generator
    # Get the latent vector
    random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
    with tf.GradientTape() as tape:
        # Generate fake sounds using the generator
        generated_sounds = self.generator(random_latent_vectors, training=True)
        # Get the discriminator logits for fake sounds
        gen_sound_logits = self.discriminator(generated_sounds, training=True)
        # Calculate the generator loss
        g_loss = self.g_loss_fn(gen_sound_logits)

    # Get the gradients w.r.t the generator loss
    gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
    # Update the weights of the generator using the generator optimizer
    self.g_optimizer.apply_gradients(
        zip(gen_gradient, self.generator.trainable_variables)
    )
    return {"d_loss": d_loss, "g_loss": g_loss}


class GANMonitor(tf.keras.callbacks.Callback):
  def __init__(self, num_sounds=6, latent_dim=256, sr = 16000):
      self.num_sounds = num_sounds
      self.latent_dim = latent_dim
      self.sr = sr

  def on_epoch_end(self, epoch, logs=None):
    random_latent_vectors = tf.random.normal(shape=(self.num_sounds, self.latent_dim))
    generated_sounds = self.model.generator(random_latent_vectors)

    for i in range(self.num_sounds):
      sound = generated_sounds[i].numpy()
      sf.write(f'..\logs\soung_generated_epoch_{epoch}_{i}.wav', sound, self.sr, 'PCM_24')

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