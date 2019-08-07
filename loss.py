import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy

def Custom_loss(model):
    recon_loss = binary_crossentropy(model.input, model.output[-1])
    kl_loss = 1 + model.output[1] - tf.square(model.output[0]) - tf.exp(model.output[1])
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    loss = tf.reduce_mean(recon_loss - 0.5*kl_loss)