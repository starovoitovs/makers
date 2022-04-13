from InitialValue import InitialValue
import tensorflow as tf


class NNsolver:
    def __init__(self, dynamic_system, explicit_time_dependence=False, optimizer=Adam, learning_rate=1e-6, loss='mse',
                 name='', n_hidden_units=None):
        self.dynamic_system = dynamic_system
        self.explicit_time_dep = explicit_time_dependence

        self.n_hidden_units = self.dynamic_system.n_dimensions + self.dynamic_system.n_diffusion_factors + self.dynamic_system.n_jump_factors + 10 \
            if n_hidden_units is not None else 10

        # Create NNs
        paths = []
        inputs_dW = Input(shapre=(self.dynamic_system.n_timesteps, self.dynamic_system.n_diffusion_factors))
        inputs_dN = Input(shape=(self.dynamic_system.n_timesteps, self.dynamic_system.n_jump_factors)) \
            if self.dynamic_system.with_jumps else None

        x0 = tf.Variable([self.dynamic_system.initX], trainable=False)
        y0 = tf.Variable([self.dynamic_system.dg(x0[0])], trainable=True)

        x = InitialValue(x0, name='x_0')(inputs_dW)
        y = InitialValue(y0, name='y_0')(inputs_dW)

        z = concatenate([x, y])
        z = Dense(n_hidden_units, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=1e-1),
                  name='z1_0')(z)
        z = Dense(self.dynamic_system.n_dimensions * self.dynamic_system.n_diffusion_factors, activation='relu',
                  kernel_initializer=initializers.RandomNormal(stddev=1e-1), name='z2_0')(z)
        z = Reshape((self.dynamic_system.n_dimensions, self.dynamic_system.n_diffusion_factors), name='zr_0')(z)

        if self.dynamic_system.with_jumps:
            r = concatenate([x, y])
            r = Dense(n_hidden_units, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=1e-1),
                      name='r1_0')(r)
            r = Dense(self.dynamic_system.n_dimensions * self.dynamic_system.n_jump_factors, activation='relu',
                      kernel_initializer=initializers.RandomNormal(stddev=1e-1), name='r2_0')(r)
            r = Reshape((self.dynamic_system.n_dimensions, self.dynamic_system.n_jump_factors), name='rr_0')(r)

            paths += [[x, y, z, r]]
        else:
            r = None  # probably redundant
            paths += [[x, y, z]]

        for i in range(self.dynamic_system.n_timesteps):
            step = InitialValue(tf.Variable(i, dtype=tf.float32, trainable=False))(inputs_dW)
            # TODO(Konstantins): I (Milo) don't really understand the functionality of the InitialValue class,
            #  particularly as it's used here

            dW = Lambda(lambda x: x[0][:, tf.cast(x[1], tf.int32)])([inputs_dW, step])
            dN = Lambda(lambda x: x[0][:, tf.cast(x[1], tf.int32)])([inputs_dN, step]) \
                if self.dynamic_system.with_jumps else None

            x, y = (
                Lambda(self.dynamic_system.hx, name=f'x_{i + 1}')([step, x, y, z, r, dW, dN]),
                Lambda(self.dynamic_system.hy, name=f'y_{i + 1}')([step, x, y, z, r, dW, dN]),
            )

            # we don't train z for the last time step; keep for consistency
            z = concatenate([x, y])
            z = Dense(n_hidden_units, activation='relu', name=f'z1_{i + 1}')(z)
            z = Dense(self.dynamic_system.n_dimensions * self.dynamic_system.n_diffusion_factors, activation='relu',
                      name=f'z2_{i + 1}')(z)
            z = Reshape((self.dynamic_system.n_dimensions, self.dynamic_system.n_diffusion_factors),
                        name=f'zr_{i + 1}')(z)

            if self.dynamic_system.with_jumps:
                # we don't train r for the last time step; keep for consistency
                r = concatenate([x, y])
                r = Dense(n_hidden_units, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=1e-1),
                          name=f'r1_{i + 1}')(r)
                r = Dense(self.dynamic_system.n_dimensions * self.dynamic_system.n_jump_factors, activation='relu',
                          kernel_initializer=initializers.RandomNormal(stddev=1e-1), name=f'r2_{i + 1}')(r)
                r = Reshape((self.dynamic_system.n_dimensions, self.dynamic_system.n_jump_factors), name=f'rr_{i + 1}')(
                    r)

                paths += [[x, y, z, r]]
            else:
                paths += [[x, y, z]]

            outputs_loss = Lambda(lambda vector: vector[1]
                                                 - tf.transpose(tf.vectorized_map(self.dynamic_system.dg, vector[0])))(
                [x, y])

            self.model_loss = Model(self.dynamic_system.array_dW_dN(inputs_dW, inputs_dN), outputs_paths)
            # TODO(milo): make write the next if/else statement cleverer/more compact. Also, it seems redudant, as
            #  nothing is done with model paths
            if self.dynamic_system.with_jumps:
                outputs_paths = tf.stack(
                    [tf.stack([p[0] for p in paths[1:]], axis=1), tf.stack([p[1] for p in paths[1:]], axis=1)] +
                    [tf.stack([p[2][:, :, i] for p in paths[1:]], axis=1) for i in
                     range(self.dynamic_system.n_diffusion_factors)] +
                    [tf.stack([p[3][:, :, i] for p in paths[1:]], axis=1) for i in
                     range(self.dynamic_system.n_jump_factors)], axis=2)
            else:
                outputs_paths = tf.stack(
                    [tf.stack([p[0] for p in paths[1:]], axis=1), tf.stack([p[1] for p in paths[1:]], axis=1)] +
                    [tf.stack([p[2][:, :, i] for p in paths[1:]], axis=1) for i in
                     range(self.dynamic_system.n_diffusion_factors)], axis=2)
            model_paths = Model(self.dynamic_system.array_dW_dN(inputs_dW, inputs_dN))
            self.optimizer = optimizer(learning_rate=learning_rate)

            self.model_loss.compile(loss=loss, optimizer=adam)

    def train(self, epochs=1000, batch_size=128, n_paths=2e6):
        dt = self.dynamic_system.dt
        dW = tf.sqrt(dt) * tf.random.normal(
            (n_paths, self.dynamic_system.n_timesteps, self.dynamic_system.n_diffusion_factors))
        if self.dynamic_system.with_jumps:
            dN = tf.random.poisson((n_paths, self.dynamic_system.n_timesteps),
                                   tf.constant(dt * np.array(
                                       [self.dynamic_system.lp, self.dynamic_system.lm]).transpose().reshape(-1)))
        else:
            dN = None
        target = tf.zeros((n_paths, self.dynamic_system.n_dimensions))

        log_dir = "_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_callback = ModelCheckpoint(f'_models/{self.name}_weights{{epoch:04d}}.h5',
                                              save_weights_only=True, overwrite=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.model_loss.save_weights('_models/weights0000.h5')
        history = self.model_loss.fit(self.dynamic_system.array_dW_dN(dW, dN), target, batch_size=batch_size,
                                      epochs=epochs, callbacks=[checkpoint_callback, tensorboard_callback])
        # TODO(konstantins) check what you want to do with `history`

    def validate(self, n_paths):
        dt = self.dynamic_system.dt
        target_test = tf.zeros((n_paths, self.dynamic_system.n_dimensions))
        dW_test = tf.sqrt(dt) * tf.random.normal((n_paths, self.dynamic_system.n_timesteps,
                                                  self.dynamic_system.n_diffusion_factors))
        if self.dynamic_system.with_jumps:
            dN_test = tf.random.poisson((n_paths, self.dynamic_system.n_timesteps),
                                        tf.constant(dt * np.array(
                                            [self.dynamic_system.lp, self.dynamic_system.lm]).transpose().reshape(-1)))
            self.model_loss.evaluate([dW_test, dN_test], target_test)
        else:
            self.model_loss.evaluate([dW_test], target_test)
