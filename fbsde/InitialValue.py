from tensorflow import Layer


class InitialValue(Layer):

    def __init__(self, y0, **kwargs):
        super().__init__(**kwargs)
        self.y0 = y0

    def call(self, inputs):
        return self.y0
