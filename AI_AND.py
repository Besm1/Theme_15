LEARNING_RATE = 0.1
LEARNING_ITER = 50

class Perceptron:

    def __init__(self, num_inputs):
        self.w = [np_random() for _ in range(num_inputs)]
        self.bias = np_random()

    def feed_forward(self, inputs, call_back_ff):
        return call_back_ff(sum(map(lambda x,y: x*y, inputs, self.w)) + self.bias)

    def train_func(self, inputs, estimation:float, call_back_ff, learn_rate):
        out = self.feed_forward(inputs, call_back_ff)
        err = estimation - out

        print('\n', self.w, self.bias)
        print(f'{inputs} --> {out}, ожидалось {estimation}. Ошибка = {err}')

        self.w = [w_ + err * inp_ * learn_rate for w_, inp_ in  zip(self.w, inputs)]  # err * inputs * LEARNING_RATE
        self.bias += err * learn_rate

        return err


def activation_function(inp):
    return 1.0 if inp >= 0.0 else 0.0


if __name__ == '__main__':
    perceptron = Perceptron(2)

    tr_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    # tr_output = [0.0, 0.0, 0.0, 1.0]      # AND
    tr_output = [0.0, 1.0, 1.0, 1.0]        # OR
    # tr_output = [0.0, 1.0, 1.0, 0.0]        # XOR

    for i in range(LEARNING_ITER):
        for inputs, estimation in zip(tr_inputs, tr_output):
            perceptron.train_func(inputs, estimation, activation_function, LEARNING_RATE)

    print(perceptron.w, perceptron.bias)

    for inp in tr_inputs:
        print(f'Inp1: {inp[0]}, Inp2: {inp[1]}, Out: {perceptron.feed_forward(inp, activation_function)}')
