import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import math
import os

DEFAULT_MODE = 1
DEFAULT_IMAGE_FILE_NAME = 'moonlanding.png'
LOW_FREQUENCY_PROPORTION = 0.1  # the lowest 10% of frequencies are considered "low"


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
    sub_problem_thresh = 2
    if len(array) <= sub_problem_thresh:
        return naiveFourierTransform(array)

    X_even = fastFourierTransform(evens(array), inverse)
    X_odds = fastFourierTransform(odds(array), inverse)

    N = len(array)

    complex_part = 2j if inverse else -2j
    divide = N if inverse else 1

    factor = lambda m: np.exp(complex_part * np.pi * m / N)

    return [divide * X_even[m] + divide * factor(m) * X_odds[m] for m in range(N // 2)] + [
        divide * X_even[m] - divide * factor(m) * X_odds[m] for m in range(N // 2)]


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


def resizeToPowerOf2(image_file_name):
    img = cv2.imread(image_file_name, cv2.IMREAD_GRAYSCALE)
    x = img.shape[1]
    x_log2 = math.log2(x)
    x_floor = 2 ** math.floor(x_log2)
    x_ceil = 2 ** math.ceil(x_log2)
    x = x_ceil if abs(x - x_floor) > abs(x - x_ceil) else x_floor

    y = img.shape[0]
    y_log2 = math.log2(y)
    y_floor = 2 ** math.floor(y_log2)
    y_ceil = 2 ** math.ceil(y_log2)
    y = y_ceil if abs(y - y_floor) > abs(y - y_ceil) else y_floor

    return cv2.resize(img, (x, y), interpolation=cv2.INTER_AREA)


def modeOne(image_file_name):
    img = resizeToPowerOf2(image_file_name)
    X = np.fft.fft2(img)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    im = ax[1].imshow(np.abs(X), norm=LogNorm(vmin=5))
    ax[0].set_title('Original image')
    ax[1].set_title('Fourier transform')
    fig.colorbar(im, ax=ax)

    plt.show()


def maskFrequencies(X, above=True, mask_thresh=1 - LOW_FREQUENCY_PROPORTION):  # mask higher: X, True, 0.1
    rows, cols = X.shape
    if above:
        X[math.floor(rows * (1 - mask_thresh)):math.ceil(rows * mask_thresh)] = 0
        X[:, math.floor(cols * (1 - mask_thresh)):math.ceil(cols * mask_thresh)] = 0
    else:
        X[0:math.floor(rows * (1 - mask_thresh))] = 0
        X[math.ceil(rows * mask_thresh):] = 0
        X[:, 0:math.floor(cols * (1 - mask_thresh))] = 0
        X[:, math.ceil(cols * mask_thresh):] = 0
    return X


def modeTwo(image_file_name):
    mask_thresh = 1 - LOW_FREQUENCY_PROPORTION
    img = resizeToPowerOf2(image_file_name)
    X = np.fft.fft2(img)
    maskFrequencies(X, above=True)
    # optionally use maskFrequencies with a copy of X
    x = np.fft.ifft2(X)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    ax[1].imshow(np.abs(x), cmap='gray', vmin=0, vmax=255)
    ax[0].set_title('Original image')
    ax[1].set_title('Denoised version')

    plt.show()


def saveMatrixAsMinimizedTxt(file_name, array, delimiter=' '):
    """
    Save complex matrix to txt file with the elements in the same formatting as np.savetxt, but with complex numbers with value equal to 0 (i.e. real and imaginary part are 0) written as a compact '0' to minimize file size.

    e.g. '(0.000000000000000000e+00+0.000000000000000000e+00j)' -> '0'
    """

    def formatComplex(x, fmt='%.18e'): return "({}{}{}j)".format(fmt % x.real, '+' if x.imag > 0 else '', fmt % x.imag)

    with open(file_name, 'w') as f:
        for row in array:
            line = delimiter.join("0" if value == 0 else formatComplex(value) for value in row)
            f.write(line + '\n')


class CompressionStrategy():  # enum
    LargestAll = 0  # threshold the coefficients’ magnitude and take only the largest percentile of them
    AllLowsAndLargestHighs = 1  # keep all the coefficients of very low frequencies as well as a fraction of the largest coefficients from higher frequencies to also filter the image at the same time


def modeThree(image_file_name, compression_strategy=CompressionStrategy.LargestAll):  # , compression_strategy):
    if compression_strategy == 0:
        print('Compression Strategy: LargestAll')
    elif compression_strategy == 1:
        print('Compression Strategy: AllLowsAndLargestHighs')
    else:
        raise IllegalArgumentError(
            "Compression strategy must be one of 'LargestAll' (0) or 'AllLowsAndLargestHighs' (1)")

    max_compression = 99
    img = resizeToPowerOf2(image_file_name)
    X = np.fft.fft2(img)

    if compression_strategy == 0:
        # flatten then sort by magnitude (see https://numpy.org/doc/stable/reference/generated/numpy.absolute.html)
        sorted_coefficients = np.sort(np.abs(X.reshape(-1)))
    elif compression_strategy == 1:
        # filter out low frequencies and flatten matrix
        high_coefficients = maskFrequencies(np.array(X), above=False).reshape(-1)
        # remove filtered (zeroed-out elements)
        high_coefficients = np.extract(high_coefficients != 0, high_coefficients)
        print(np.count_nonzero(high_coefficients))
        # sort by magnitude
        sorted_coefficients = np.sort(np.abs(high_coefficients))

    coefficient_count = len(sorted_coefficients)
    print(coefficient_count)

    fig, ax = plt.subplots(2, 3)
    for i, compression_level in enumerate(np.linspace(0, max_compression, num=6, dtype=np.int_)):
        threshold = sorted_coefficients[int(compression_level * coefficient_count / 100)]

        mask = np.abs(
            X) >= threshold  # matrix with the same shape as X with 1s where the magnitude of the coefficients are >= threshold and 0s elsewhere
        X_largest_coefficients = mask * X  # mask the coefficients with magnitudes < threshold

        if compression_strategy == 0:
            compressed_X = X_largest_coefficients
        elif compression_strategy == 1:
            X_low_frequencies = maskFrequencies(np.array(X))
            compressed_X = np.where(X_low_frequencies != 0, X_low_frequencies, X_largest_coefficients)

        file_name = str(compression_level) + '%.txt'
        saveMatrixAsMinimizedTxt(file_name, compressed_X)

        compressed_image = np.abs(np.fft.ifft2(compressed_X))

        title = 'Original' if i == 0 else str(compression_level) + '%'
        ax[i // 3, i % 3].imshow(compressed_image, cmap='gray', vmin=0, vmax=255)
        ax[i // 3, i % 3].title.set_text(title)

        size = os.stat(file_name)
        print(
            'Compression: {compression_level}% | Non-Zero Values: {values} | File Size (bytes): {size}'.format(
                compression_level=compression_level,
                values=(100 - compression_level) * coefficient_count,
                size=size.st_size
            )
        )
    plt.show()


def modeFour():
    EXPERIMENTS = 10
    MIN_PROBLEM_SIZE_POWER = 5
    max_problem_size_power = 7  # 2^7 takes ~10s
    data = {}
    total_start = time.perf_counter()
    for i in range(MIN_PROBLEM_SIZE_POWER, max_problem_size_power):
        size = 2 ** i
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

    total_end = time.perf_counter()
    print("finished experiments in {}s\n".format(total_end - total_start))

    sorted_sizes = sorted(data.keys())

    naive_means = [np.mean(data[size]["naive"]) for size in sorted_sizes]
    fast_means = [np.mean(data[size]["fast"]) for size in sorted_sizes]

    naive_stds = [np.std(data[size]["naive"]) for size in sorted_sizes]
    fast_stds = [np.std(data[size]["fast"]) for size in sorted_sizes]

    print("Experiments per problem size: {}\nConfidence Interval: {}%\n".format(EXPERIMENTS, 97))

    # plot
    labels = sorted_sizes
    x = np.arange(len(labels))

    fig, ax = plt.subplots()

    width = 0.35  # the width of the bars
    capsize = 5
    naive_bars = ax.bar(x - width / 2, naive_means, yerr=[2 * std for std in naive_stds], width=width, label="Naïve",
                        capsize=capsize)
    fast_bars = ax.bar(x + width / 2, fast_means, yerr=[2 * std for std in fast_stds], width=width, label="Fast",
                       capsize=capsize)

    ax.set_title("Runtimes by problem size for naïve and fast Fourier transform")
    ax.set_xlabel("Problem size")
    ax.set_ylabel("Runtime (s)")
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # ax.bar_label(naive_bars, padding=3)
    # ax.bar_label(fast_bars, padding=3)

    fig.tight_layout()

    for i, size in enumerate(sorted_sizes):
        print("Problem size: {}x{}".format(size, size))
        print("Naive: mean={}, variance={}".format(naive_means[i], naive_stds[i]))
        print("Fast: mean={}, variance={}".format(fast_means[i], fast_stds[i]))
        print()

    plt.show()


def printError(message, prefix="ERROR"):
    print(prefix + "\t" + message)


def getParams():
    parser = argparse.ArgumentParser(
        exit_on_error=False)  # if there is an error raised by this line, please consult the README.md
    parser.add_argument("-m", "--mode", type=int, help="mode (optional)", default=DEFAULT_MODE)
    parser.add_argument("-i", "--image", type=str,
                        help="image (optional) filename of the image we wish to take the DFT of",
                        default=DEFAULT_IMAGE_FILE_NAME)

    try:
        namespace, rest = parser.parse_known_args()  # https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_known_args

        for arg in rest:
            raise IllegalArgumentError("Incorrect input syntax: '{}' argument is not recognized".format(arg))

        params = vars(namespace)

        if params["mode"] < 1 or params["mode"] > 4:
            raise IllegalArgumentError("Incorrect input syntax: mode must be between 1 and 4")

        params["image"] = params["image"].strip()

        return params

    except argparse.ArgumentError as e:
        raise IllegalArgumentError("Incorrect input syntax: " + str(e))


def main():
    mode = 0
    image_file_name = ""

    try:
        params = getParams()
        mode = params["mode"]
        image_file_name = params["image"]
    except IllegalArgumentError as e:
        printError(str(e))

    print("Mode " + str(mode) + "\n")

    if mode == 1:
        modeOne(image_file_name)
    elif mode == 2:
        modeTwo(image_file_name)
    elif mode == 3:
        try:
            modeThree(image_file_name)
        except IllegalArgumentError as e:
            printError(str(e))
    elif mode == 4:
        modeFour()


if __name__ == '__main__':
    main()
