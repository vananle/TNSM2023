import os.path
from os import path

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print('Invalid device or cannot modify virtual devices once initialized')
    pass


class Sampling(tf.keras.layers.Layer):
    """
    (1) Sampling Layer is a subclass of tf.keras.layers.Layer
    (2) Reparameterization trick: use z_mean and z_log_var to sample z
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.math.exp(0.5 * z_log_var) * epsilon


class VAE(tf.keras.Model):
    """
    Our VAE is a subclass of tf.keras.Model
    """

    def __init__(self, args, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.args = args

        self.X_val = None
        self.latent_dim = self.args.latent_dim
        self.num_node = self.args.num_node

        self.encoder = None
        self.decoder = None

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.total_val_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

        self.max_optimization_iterations = 5000
        self.max_initial_point_iterations = 500

        self.save_path = None

    def set_X_val(self, X_val):

        self.X_val = tf.convert_to_tensor(X_val)

    def set_save_path(self, save_path):
        self.save_path = save_path

    def initializing(self):

        if self.args.test:
            load_saved_model = True
        else:
            load_saved_model = False

        if not load_saved_model:

            # Encoder
            encoder_inputs = tf.keras.Input(shape=(self.num_node, self.num_node, 1))
            x = tf.keras.layers.Dropout(0.25)(encoder_inputs)
            x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
            x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
            x = tf.keras.layers.Conv2D(128, 3, activation="relu", strides=1, padding="same")(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(64, activation="relu")(x)
            z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean")(x)
            z_log_var = tf.keras.layers.Dense(self.latent_dim, name="z_log_var")(x)
            z = Sampling()([z_mean, z_log_var])
            encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
            # encoder.summary()

            # Decoder
            latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
            x = tf.keras.layers.Dense(64, activation="relu")(latent_inputs)
            x = tf.keras.layers.Dense(3 * 3 * 64, activation="relu")(x)
            x = tf.keras.layers.Reshape((3, 3, 64))(x)
            x = tf.keras.layers.Conv2DTranspose(128, 3, activation="relu", strides=1, padding="same")(x)

            size = 3
            hidden = 64
            while size < self.args.num_node:
                x = tf.keras.layers.Conv2DTranspose(hidden, 3, activation="relu", strides=2, padding="same")(x)
                hidden = int(hidden / 2) if hidden > 2 else 1
                size = size * 2

            if hidden > 1:
                x = tf.keras.layers.Conv2DTranspose(1, 3, activation="relu", padding="same")(x)

            x = tf.keras.layers.Reshape((size * size * 1,))(x)
            x = tf.keras.layers.Dense(self.args.num_node * self.args.num_node * 1, activation="relu")(x)
            decoder_outputs = tf.keras.layers.Reshape((self.args.num_node, self.args.num_node, 1))(x)
            decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
            # decoder.summary()

            self.encoder = encoder
            self.decoder = decoder
        else:

            print('Load saved model')
            self.encoder = tf.keras.models.load_model(os.path.join(self.save_path, 'encoder'))
            self.decoder = tf.keras.models.load_model(os.path.join(self.save_path, 'decoder'))

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.total_val_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

        self.max_optimization_iterations = 5000
        self.max_initial_point_iterations = 500

    @property
    def metrics(self):
        """
        Model calls automatically reset_states() on any object listed here,
        at the beginning of each fit() epoch or at the begining of a call to evaluate().
        In this way, calling result() would return per-epoch average and not an average
        since the start of training.
        """
        return [self.total_loss_tracker]

    def train_step(self, x_true):
        """
        (1) Override train_step(self, x_true) to customize what fit() does.
        (2) We use GradientTape() in order to record operations for automatic differentiation.
        """
        x_val = self.X_val
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x_true)
            x_pred = self.decoder(z)
            # reconstruction loss: mean squared error
            reconstruction_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(x_true, x_pred), axis=(1, 2))
            # regularization term: KL divergence
            kl_loss = tf.reduce_sum(-0.5 * (1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)), axis=1)
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

        z_mean_val, z_mean_log_var, z_val = self.encoder(x_val)
        x_val_pred = self.decoder(z_val)
        # reconstruction loss: mean squared error
        reconstruction_val_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(x_val, x_val_pred), axis=(1, 2))
        # regularization term: KL divergence
        kl_val_loss = tf.reduce_sum(
            -0.5 * (1 + z_mean_log_var - tf.math.square(z_mean_val) - tf.math.exp(z_mean_log_var)), axis=1)
        total_val_loss = tf.reduce_mean(reconstruction_val_loss + kl_val_loss)

        # self.trainable_variables and self.optimizer are inherited from tf.keras.Model
        # Get gradients of total loss with respect to the weights.
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Update the weights of the model.
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.total_val_loss_tracker.update_state(total_val_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "val_loss": self.total_val_loss_tracker.result()
        }
