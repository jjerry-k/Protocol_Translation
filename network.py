import tensorflow as tf
from tensorflow.keras import layers, models

def unet(use_bias=True):
    _input = layers.Input(shape=(None, None, 1, ))
    en1 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(_input)
    en1 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(en1)
    
    en2 = layers.MaxPool2D()(en1)
    en2 = layers.Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(en2)
    en2 = layers.Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(en2)
    
    en3 = layers.MaxPool2D()(en2)
    en3 = layers.Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(en3)
    en3 = layers.Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(en3)
    
    en4 = layers.MaxPool2D()(en3)
    en4 = layers.Conv2D(512, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(en4)
    en4 = layers.Conv2D(512, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(en4)
    
    en5 = layers.MaxPool2D()(en4)
    en5 = layers.Conv2D(1024, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(en5)
    en5 = layers.Conv2D(1024, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(en5)
    
    de4 = layers.Conv2DTranspose(512, 2, strides=(2,2), padding='same', activation='relu', use_bias=use_bias)(en5)
    de4 = layers.Concatenate(axis=3)([en4, de4])
    de4 = layers.Conv2D(512, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(de4)
    de4 = layers.Conv2D(512, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(de4)
    
    de3 = layers.Conv2DTranspose(256, 2, strides=(2,2), padding='same', activation='relu', use_bias=use_bias)(de4)
    de3 = layers.Concatenate(axis=3)([en3, de3])
    de3 = layers.Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(de3)
    de3 = layers.Conv2D(256, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(de3)
    
    de2 = layers.Conv2DTranspose(128, 2, strides=(2,2), padding='same', activation='relu', use_bias=use_bias)(de3)
    de2 = layers.Concatenate(axis=3)([en2, de2])
    de2 = layers.Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(de2)
    de2 = layers.Conv2D(128, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(de2)
    
    de1 = layers.Conv2DTranspose(64, 2, strides=(2,2), padding='same', activation='relu', use_bias=use_bias)(de2)
    de1 = layers.Concatenate(axis=3)([en1, de1])
    de1 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(de1)
    de1 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(de1)
    
    pred = layers.Conv2D(1, 1, strides=(1, 1), padding='same', activation='relu', use_bias=use_bias)(de1)
    return models.Model(inputs = _input, outputs=pred)


def z_sampling(argv):
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.random_normal(shape=tf.shape(argv[0]))
    return argv[0] + tf.exp(0.5 * argv[1]) * epsilon

def VAE(latent_dim = 16, last_activation = 'relu',use_bias=True):
    # h x w
    _x = layers.Input(shape=(None, None, 1), name="Input")
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=use_bias, name="Encoder_1")(_x)
    x = layers.MaxPool2D()(x)
    # h/2 x w/2
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu', use_bias=use_bias, name="Encoder_2")(x)
    x = layers.MaxPool2D()(x)
    # h/4 x w/4
    
    x = layers.Conv2D(256, 3, padding='same', activation='relu', use_bias=use_bias, name="Encoder_3")(x)
    x = layers.MaxPool2D()(x)
    # h/8 x w/8
    
    x = layers.Conv2D(512, 3, padding='same', activation='relu', use_bias=use_bias, name="Encoder_4")(x)
    x = layers.MaxPool2D()(x)
    # h/16 x w/16
    
    x = layers.Conv2D(1024, 3, padding='same', activation='relu', use_bias=use_bias, name="Encoder_5")(x)
    
    # Spatial z_mean, z_log_var
    mean = layers.Conv2D(latent_dim, 1, name="z_mean")(x)
    log_var = layers.Conv2D(latent_dim, 1, name="z_log_var")(x)
    
    z = layers.Lambda(z_sampling)([mean, log_var])
    
    # h/8 x w/8
    x = layers.UpSampling2D(interpolation='bilinear')(z)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', use_bias=use_bias, name="Decoder_4")(x)
    
    # h/4 x w/4
    x = layers.UpSampling2D(interpolation='bilinear')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', use_bias=use_bias, name="Decoder_3")(x)
    
    # h/2 x w/2
    x = layers.UpSampling2D(interpolation='bilinear')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', use_bias=use_bias, name="Decoder_2")(x)
    
    # h x w
    x = layers.UpSampling2D(interpolation='bilinear')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=use_bias, name="Decoder_1")(x)
    
    # Prediction
    x = layers.Conv2D(1, 1, padding='same', activation=last_activation, use_bias=use_bias, name="Prediction")(x)
    
    return models.Model(inputs=_x, outputs = [mean, log_var, x])

