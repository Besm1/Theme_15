from random import random as np_random

class Perceptron:

    def __init__(self, num_inputs):
        self.w = [np_random() for _ in range(num_inputs)]
        self.bias = np_random()

    def feed_forward(self, inputs, call_back_ff):
        return call_back_ff(sum(map(lambda x,y: x*y, inputs, self.w)) + self.bias)

    def train_func(self, inputs, estimation:float, call_back_ff, learn_rate):
        out = self.feed_forward(inputs, call_back_ff)
        err = estimation / out

        print('\n', self.w, self.bias)
        print(f'{inputs} --> {out}, ожидалось {estimation}. Ошибка = {err}')

        self.w = [w_ + err * inp_ * learn_rate for w_, inp_ in  zip(self.w, inputs)]  # err * inputs * LEARNING_RATE
        self.bias += err * learn_rate

        return err
