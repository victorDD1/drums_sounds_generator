######################
###                ###
##  TRAINING STEPS  ##
###                ###
######################


import tensorflow as tf
import time
import os

@tf.function
def train_step(samples,
               generator,
               generator_loss,
               generator_optimizer,
               discriminator,
               discriminator_loss,
               discriminator_optimizer,
               batch_size,
               noise_dim):
    
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_samples = generator(noise, training=True)

      real_output = discriminator(samples, training=True)
      fake_output = discriminator(generated_samples, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset,
          epochs,
          batch_size,
          noise_dim,
          generator,
          generator_loss,
          generator_optimizer,
          discriminator,
          discriminator_loss,
          discriminator_optimizer,
          ):
    
    checkpoint_dir = './logs/checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    for epoch in range(epochs):
        start = time.time()

        for samples_batch in dataset:
            train_step(samples_batch,
                    generator,
                    generator_loss,
                    generator_optimizer,
                    discriminator,
                    discriminator_loss,
                    discriminator_optimizer,
                    batch_size,
                    noise_dim)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))