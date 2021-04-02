import numpy as np
import argparse
import time
import matplotlib


DEFAULT_IMAGE_FILE_NAME = 'moonlanding.png'
SUB_PROBLEM_SIZE_TRESH  = 5
EXPERIMENTS = 10
CONFIDENCE_INTERVAL = 0.97


def naiveFourierTransform(array, inverse=False):
    complex_part = 2j if inverse else -2j
    divide = len(array) if inverse else 1
    result = [sum([
        x * np.exp(complex_part * np.pi * k * n / len(array)) for n, x in enumerate(array)
    ]) / divide for k, _ in enumerate(array)]
    return result


def _computeSum(array, exponent):
    return sum([
        x * exponent(m) for m, x in enumerate(array)
    ])


def _fastFourierTransform(array, exponent):
    if len(array) <= SUB_PROBLEM_SIZE_TRESH:
        return _computeSum(array, exponent(len(array)))
        
    return _fastFourierTransform(array[0:][::2], exponent) + \
        exponent(len(array))(1) * _fastFourierTransform(array[1:][::2], exponent)


def fastFourierTransform(array, inverse=False):
    complex_part = 2j if inverse else -2j
    divide = len(array) if inverse else 1
    return [
        divide * _fastFourierTransform(array, (lambda N: lambda m: np.exp(complex_part * np.pi * k * m / N)))
        for k in range(len(array))
    ]


def naiveFourierTransformMatrix(matrix, inverse=False):
    matrix = np.array(matrix)
    row_result = np.array([
        naiveFourierTransform(row, inverse)
        for row in matrix
    ])
    return np.array([
        naiveFourierTransform(row_result[:, col], inverse)
        for col in range(len(row_result[0]))
    ]).T


def fastFourierTransformMatrix(matrix, inverse=False):
    matrix = np.array(matrix)
    row_result = np.array([
        fastFourierTransform(row, inverse)
        for row in matrix
    ])
    return np.array([
        fastFourierTransform(row_result[:, col], inverse)
        for col in range(len(row_result[0]))
    ]).T

# if __name__ == '__main__':
#     pass

def generateRandomSquareMatrix(size):
    return np.random.rand(size, size)

def modeFour():
    data = {}

    for i in range(10):
        size = 2**i
        data[size] = {}

        data[size]["naive"] = []
        data[size]["fast"] = []

        for j in range(EXPERIMENTS):
            random_matrix = generate_random_square_matrix(size)

            start = time.perf_counter()
            naiveFourierTransformMatrix(random_matrix)
            end = time.perf_counter()

            data[size]["naive"].append(end - start)
            
            start = time.perf_counter()
            fastFourierTransformMatrix(random_matrix)
            end = time.perf_counter()

            data[size]["fast"].append(end - start)

