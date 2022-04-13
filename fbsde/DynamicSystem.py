class DynamicSystem:
    def __init__(self, b, sigma, v, min_hamil_deriv_X, dg, initX, n_timesteps=100, T=1, lp=None, lm=None):
        """
        All functions take input t, x, y, z, r
        v can be None
        """
        self.b = b
        self.sigma = sigma
        self.v = v
        self.min_hamil_deriv_X = min_hamil_deriv_X
        self.dg = dg

        self.initX = initX
        self.n_dimensions = np.size(initX)

        self.n_timesteps = n_timesteps
        self.T = T
        self.dt = T / self.dynamic_system.n_timesteps

        self.with_jumps = True if self.v is not None else False

        self.n_diffusionfactors = len(sigma(0, 0, 0, 0, 0))  # maybe have to add [0]

        # TODO(Milo): check how to handle/generalize these lp and lm
        self.lp = lp
        self.lm = lm

        if self.with_jumps:
            self.n_jump_factors = len(v(0, 0, 0, 0, 0))
            assert not ((self.lp is None) and (self.lm is None)), "No jumping intensity specified"
        else:
            self.n_jump_factors = 0

    def dX(self, t, x, y, z, r, dW, dN):
        """one step in X"""
        def drift(arg):
            x, y, z, r = arg
            return tf.math.multiply(self.b(t, x, y, z, r), dt)

        a0 = tf.vectorized_map(drift, (x, y, z, r))

        def noise(arg):
            x, y, z, r, dW = arg
            return tf.tensordot(self.sigma(t, x, y, z, r), dW, [[1], [0]])

        a1 = tf.vectorized_map(noise, (x, y, z, r, dW))

        def jump(arg):
            x, y, z, r, dN = arg
            if not self.with_jumps:
                return tf.zeros(shape=tf.shape(x))
            return tf.tensordot(self.v(t, x, y, z, r), dN, [[1], [0]])

        a2 = tf.vectorized_map(jump, (x, y, z, r, dN))

        return a0 + a1 + a2

    def dY(self, t, x, y, z, r, dW, dN):
        """One step in Y"""
        def drift(arg):
            x, y, z, r = arg
            return tf.math.multiply(self.min_hamil_deriv_X(t, x, y, z, r), dt)

        a0 = tf.vectorized_map(drift, (x, y, z, r))

        def noise(arg):
            x, y, z, r, dW = arg
            return tf.tensordot(z, dW, [[1], [0]])

        a1 = tf.vectorized_map(noise, (x, y, z, r, dW))

        def jump(arg):
            x, y, z, r, dN = arg
            return tf.tensordot(r, dN, [[1], [0]])

        a2 = tf.vectorized_map(jump, (x, y, z, r, dN))

        return a0 + a1 + a2

    @tf.function
    def hx(self, args):
        i, x, y, z, r, dW, dN = args
        return x + self.dX(i * dt, x, y, z, r, dW, dN)

    @tf.function
    def hy(self, args):
        i, x, y, z, r, dW, dN = args
        return y + self.dY(i * dt, x, y, z, r, dW, dN)

    def array_dW_dN(self, dW, dN):
        if self.with_jumps:
            return [dW, dN]
        return [dW]
