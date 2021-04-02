import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import math


DEFAULT_MODE = 1
DEFAULT_IMAGE_FILE_NAME = 'moonlanding.png'
SUB_PROBLEM_SIZE_TRESH  = 5


class IllegalArgumentError(ValueError):
    pass


def naiveFourierTransform(array, inverse=False):
    complex_part = 2j if inverse else -2j
    divide = len(array) if inverse else 1
    result = [sum([
        x * np.exp(complex_part * np.pi * k * n / len(array)) for n, x in enumerate(array)
    ]) / divide for k in range(len(array))]
    return result


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

    return [divide * X_even[m] + divide * factor(m) * X_odds[m] for m in range(N//2)] + [divide * X_even[m] - divide * factor(m) * X_odds[m] for m in range(N//2)] 


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

def resizeToPowerOf2(imageFileName):
    img = cv2.imread(imageFileName, cv2.IMREAD_GRAYSCALE)
    x = img.shape[1]
    x_log2 = math.log2(x)
    x_floor = 2**math.floor(x_log2)
    x_ceil = 2**math.ceil(x_log2)
    x = x_ceil if abs(x - x_floor) > abs(x-x_ceil) else x_floor

    y = img.shape[0]
    y_log2 = math.log2(y)
    y_floor = 2**math.floor(y_log2)
    y_ceil = 2**math.ceil(y_log2)
    y = y_ceil if abs(y - y_floor) > abs(y - y_ceil) else y_floor

    return cv2.resize(img, (x, y), interpolation = cv2.INTER_AREA)


def modeOne(imageFileName):
    img = resizeToPowerOf2(imageFileName)
    X = np.fft.fft2(img)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    ax[1].imshow(np.abs(X), norm=LogNorm(vmin=5))
    plt.title('Fourier transform')
    plt.show()


def modeTwo(imageFileName):
    img = resizeToPowerOf2(imageFileName)
    X = np.fft.fft2(img)
    rows, cols = X.shape
    X[int(rows*0.1):int(rows*0.9)] = 0
    X[:,int(cols * 0.1):int(cols * 0.9)] = 0
    X = np.fft.ifft2(X)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    ax[1].imshow(np.abs(X), cmap='gray', vmin=0, vmax=255)
    plt.title('Fourier transform')
    plt.show()

def modeFour():
    EXPERIMENTS = 10
    MAX_MATRIX_SIZE_POWER = 6 # 2^7 takes ~10s
    CONFIDENCE_INTERVAL = 0.97 # idk
    data = {}

    for i in range(MAX_MATRIX_SIZE_POWER):
        size = 2**i
        data[size] = {"naive": [], "fast": []}

        for j in range(EXPERIMENTS):
            random_matrix = np.random.rand(size, size)  # random size x size matrix

            start = time.perf_counter()
            naiveFourierTransformMatrix(random_matrix)
            end = time.perf_counter()
            data[size]["naive"].append(end - start)
            
            start = time.perf_counter()
            fastFourierTransformMatrix(random_matrix)
            end = time.perf_counter()
            data[size]["fast"].append(end - start)

    sorted_sizes = sorted(data.keys())

    naive_means = [np.mean(data[size]["naive"]) for size in sorted_sizes]
    fast_means = [np.mean(data[size]["fast"]) for size in sorted_sizes]

    naive_stds = [np.std(data[size]["naive"]) for size in sorted_sizes]
    fast_stds = [np.std(data[size]["fast"]) for size in sorted_sizes]

    print("Experiments: {}\nConfidence Interval: {}%".format(EXPERIMENTS, CONFIDENCE_INTERVAL*100))

    # plot
    labels = sorted_sizes
    x = np.arange(len(labels))

    fig, ax = plt.subplots()

    width = 0.35  # the width of the bars
    capsize = 5
    naive_bars = ax.bar(x - width/2, naive_means, yerr=naive_stds, width=width, label="Naïve", capsize=capsize)
    fast_bars = ax.bar(x + width/2, fast_means, yerr=fast_stds, width=width, label="Fast", capsize=capsize)

    ax.set_title("Runtimes by problem size for naïve and fast Fourier transform")
    ax.set_xlabel("Problem size")
    ax.set_ylabel("Runtime (s)")
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # ax.bar_label(naive_bars, padding=3)
    # ax.bar_label(fast_bars, padding=3)

    fig.tight_layout()
    plt.show()


def printError(message, prefix="ERROR"):
    print(prefix + "\t" + message)


def getParams():
    parser = argparse.ArgumentParser(exit_on_error=False) # if there is an error raised by this line, please consult the README.md
    parser.add_argument("-m", "--mode", type=int, help="mode (optional)", default=DEFAULT_MODE)
    parser.add_argument("-i", "--image", type=str, help="image (optional) filename of the image we wish to take the DFT of", default=DEFAULT_IMAGE_FILE_NAME)
    
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
    
    print("Mode " + str(mode) + "\n")

    if mode ==  1:
        modeOne(imageFileName)
    elif mode == 2:
        modeTwo(imageFileName)
    elif mode == 4:
        modeFour()


if __name__ == '__main__':
    main()
