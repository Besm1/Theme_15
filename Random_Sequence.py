from random import randint

from sympy import sequence

LEARNING_RATE = 1
SEQUENCE_LENGTH = 25
TRAIN_CICLE = 15
INPUTS_QTY = 3

from random import random as np_random

def fib(n) -> list:
    if n == 2:
        return [1, 1]
    else:
        res = fib(n-1)
        res.append(res[n-2] + res[n-3])
        return res




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

        self.w = [w_ * err  * learn_rate for w_ in  self.w]  # err * inputs * LEARNING_RATE
        self.bias *= err * learn_rate

        return err


def activation_function(inp):
    return inp if inp >= 0 else 0


if __name__ == '__main__':
    perceptron = Perceptron(INPUTS_QTY)

    # sequence = [randint(1, 256) for _ in range(SEQUENCE_LENGTH)]
    sequence = fib(SEQUENCE_LENGTH)
    print(f'Последовательность: {sequence}')


    # Training
    print('Обучение...')
    tr_error = 1
    for i in range(TRAIN_CICLE):
        # for _ in range(TRAIN_CICLE):
        tr_error *= perceptron.train_func(sequence[i : i + INPUTS_QTY], sequence[i + INPUTS_QTY]
                                          , activation_function, LEARNING_RATE)
    tr_error_avg = tr_error ** (1 / TRAIN_CICLE) # / TRAIN_CICLE

    print(f'Веса = {perceptron.w}, биас = {perceptron.bias}. Ошибка обучения = {tr_error_avg}\n')

    print('Проверка...')
    check_err = 1
    for i in range(TRAIN_CICLE, SEQUENCE_LENGTH - INPUTS_QTY):
        res = perceptron.feed_forward(sequence[i : i + INPUTS_QTY], activation_function)
        err = sequence[i + INPUTS_QTY] / res
        check_err *= err
        print(f'{sequence[i : i + INPUTS_QTY ]} --> {res}, ожидалось {sequence[i + INPUTS_QTY]}. Ошибка = {err}')

    print(f'\nСредняя ошибка проверки = {check_err ** (1 / (SEQUENCE_LENGTH - TRAIN_CICLE - INPUTS_QTY))}')
