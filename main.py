import numpy as np
import argparse

DEFAULT_MODE = 1

class IllegalArgumentError(ValueError):
    pass

DEFAULT_IMAGE_FILE_NAME = 'moonlanding.png'
SUB_PROBLEM_SIZE_TRESH  = 5


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

def evens(array): return array[::2]
def odds(array): return array[1::2]

def fastFourierTransform(array, inverse=False):
    if len(array) <= SUB_PROBLEM_SIZE_TRESH:
        return naiveFourierTransform(array)

    X_even = fastFourierTransform(evens(array), inverse)
    X_odds = fastFourierTransform(odds(array), inverse)

    N = len(array)

    complex_part = 2j if inverse else -2j
    divide = N if inverse else 1
    
    factor = lambda m: np.exp(complex_part * np.pi * m / N) 
    # factor = np.exp(complex_part * np.pi * np.arange(N) / N)

    # return np.concatenate([X_even + factor[:N // 2] * X_odd, X_even + factor[N // 2:] * X_odd]) 
    return [X_even[m] + factor(m) * X_odds[m] for m in range(N//2)] + [X_even[m] - factor(m) * X_odds[m] for m in range(N//2)] 


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

def printError(message, prefix="ERROR"):
    print(prefix + "\t" + message)

def getParams():
    parser = argparse.ArgumentParser(description='Use this command line tool to query a DNS server', exit_on_error=False) # if there is an error raised by this line, please consult the README.md
    parser.add_argument("-m", "--mode", type=int, help="mode (optional)", default=DEFAULT_MODE)
    parser.add_argument("-i", "--image", type=str, help="image (optional) filename of the image we wish to take the DFT of", default='moonlanding.png')
    
    try:
        namespace, rest = parser.parse_known_args() # https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_known_args

        fatals = 0
        for arg in rest:
            raise IllegalArgumentError("Incorrect input syntax: '{}' argument is not recognized".format(arg))

        params = vars(namespace)

        if (params["mode"] < 1 or params["mode"] > 4):
            raise IllegalArgumentError("Incorrect input syntax: mode must be between 1 and 4")
        
        params["image"] = params["image"].strip()
        
        return params

    except argparse.ArgumentError as e:
        raise IllegalArgumentError("Incorrect input syntax: " + str(e))
        # assume fatal

def main():
    mode = 0
    imageFileName = ""
    try:
        params = getParams()
        mode = params["mode"]
        imageFileName = params["image"]
    except IllegalArgumentError as e:
        printError(str(e))
    

    x = np.random.random(1024)
    print(np.allclose(fastFourierTransform(x), np.fft.fft(x)))



if __name__ == '__main__':
    main()