import fft  # uut
import numpy as np
import time
import math

RELATIVE_TOLERANCE = 1e-05
ABSOLUTE_TOLERANCE = 1e-08


def randomFloatArray(size):
    if type(size) != int:
        raise fft.IllegalArgumentError("Array size must be a single integer")
    return np.random.rand(size)


def randomFloatMatrix(size):  # (rows, cols)
    if type(size) != tuple or len(size) != 2 or type(size[0]) != int or type(size[1]) != int:
        raise fft.IllegalArgumentError("Matrix size must be a tuple of two integers")
    return np.random.rand(size[0], size[1])


def randomTransformArray(size):
    return np.fft.fft(randomFloatArray(size))


def randomTransformMatrix(size):
    return np.fft.fft2(randomFloatMatrix(size))


def compareArrays(a, b):
    return np.allclose(a, b, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)


def compareMatrices(A, B):
    return np.allclose(A, B, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)


def naiveFourierTransform_TransformAnyArray_CloseToActual(array):
    X = fft.naiveFourierTransform(array, inverse=False)
    assert compareArrays(X, np.fft.fft(array))


def fastFourierTransform_TransformPowerOf2SizedArray_CloseToActual(array):
    assert fft.isPowerOf2(len(array))
    X = fft.fastFourierTransform(array, inverse=False)
    assert compareArrays(X, np.fft.fft(array))


def fastFourierTransform_TransformNonPowerOf2SizedArray_ErrorRaised(array):
    assert not fft.isPowerOf2(len(array))
    try:
        fft.fastFourierTransform(array, inverse=False)
        assert False
    except fft.IllegalArgumentError as e:
        assert str(e) == "Input vector size must be a power of 2"
        return
    assert False


def naiveFourierTransformMatrix_TransformAnyMatrix_CloseToActual(matrix):
    X = fft.naiveFourierTransformMatrix(matrix, inverse=False)
    assert compareMatrices(X, np.fft.fft2(matrix))


def fastFourierTransformMatrix_TransformPowerOf2SizedMatrix_CloseToActual(matrix):
    matrix = np.array(matrix)
    row, col = matrix.shape
    assert fft.isPowerOf2(row) and fft.isPowerOf2(col)
    X = fft.fastFourierTransformMatrix(matrix, inverse=False)
    assert compareMatrices(X, np.fft.fft2(matrix))


def fastFourierTransformMatrix_TransformNonPowerOf2SizedMatrix_ErrorRaised(matrix):
    matrix = np.array(matrix)
    row, col = matrix.shape
    assert not fft.isPowerOf2(row) and not fft.isPowerOf2(col)
    try:
        fft.fastFourierTransform(matrix, inverse=False)
        assert False
    except fft.IllegalArgumentError as e:
        assert str(e) == "Input matrix sizes must be powers of 2"
        return
    assert False


def naiveFourierTransform_InverseTransformAnyArray_CloseToActual(array):
    X = fft.naiveFourierTransform(array, inverse=True)
    assert compareArrays(X, np.fft.ifft(array))


def fastFourierTransform_InverseTransformPowerOf2SizedArray_CloseToActual(array):
    assert fft.isPowerOf2(len(array))
    X = fft.fastFourierTransform(array, inverse=True)
    assert compareArrays(X, np.fft.ifft(array))


def fastFourierTransform_InverseTransformNonPowerOf2SizedArray_ErrorRaised(array):
    assert not fft.isPowerOf2(len(array))
    try:
        fft.fastFourierTransform(array, inverse=True)
        assert False
    except fft.IllegalArgumentError as e:
        assert str(e) == "Input vector size must be a power of 2"
        return
    assert False


def naiveFourierTransformMatrix_InverseTransformAnyMatrix_CloseToActual(matrix):
    X = fft.naiveFourierTransformMatrix(matrix, inverse=True)
    assert compareMatrices(X, np.fft.ifft2(matrix))


def fastFourierTransformMatrix_InverseTransformPowerOf2SizedMatrix_CloseToActual(matrix):
    matrix = np.array(matrix)
    row, col = matrix.shape
    assert fft.isPowerOf2(row) and fft.isPowerOf2(col)
    X = fft.fastFourierTransformMatrix(matrix, inverse=True)
    assert compareMatrices(X, np.fft.ifft2(matrix))


def fastFourierTransformMatrix_TransformNonPowerOf2SizedMatrix_ErrorRaised(matrix):
    matrix = np.array(matrix)
    row, col = matrix.shape
    assert not fft.isPowerOf2(row) and not fft.isPowerOf2(col)
    try:
        fft.fastFourierTransform(matrix, inverse=True)
        assert False
    except fft.IllegalArgumentError as e:
        assert str(e) == "Input matrix sizes must be powers of 2"
        return
    assert False


def executeAndReport(test_function, data):
    """
    Expects 'data' to be a numpy array.
    Expects 'test' to be a one parameter method that accepts data as its parameter.
    """

    def formatShape(ndim, shape):
        return "length {}".format(shape[0]) if ndim == 1 else "shape {}x{}".format(shape[0], shape[1])

    print("Executing '{}' with {} of {}".format(test_function.__name__, 'array' if data.ndim == 1 else 'matrix',
                                                formatShape(data.ndim, data.shape)))

    start = time.perf_counter()
    passed = False
    error = None
    try:
        test_function(data)
        passed = True
    except Exception as e:
        passed = False
        error = e
    end = time.perf_counter()

    if passed:
        print("Passed.", end="")
    else:
        print("Failed with {} '{}'.".format(error.__class__.__name__, str(error)), end="")
    print(" (finished in {}s)".format(end - start))

    return 0 if passed else 1


def printSummary(runs, failures):
    print("{}/{}\n".format(runs - failures, runs))


def driver():
    # size limits
    naive_size_thresh = 1400  # ~1500
    fast_size_thresh = 40000

    # test data
    array_any_sizes = (1, 2, 3, 5, 8, 13, 21, 34, 50, 100, 150, 300, 600, 1200, 1400, 3000, 7000, 10000)
    array_power_of_2_sizes = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
    matrix_any_sizes = (
    (1, 1), (2, 2), (1, 2), (2, 1), (1, 3), (3, 1), (1, 2), (2, 1), (5, 9), (12, 17), (27, 69), (45, 19), (5, 14),
    (5235, 4362), (7890, 7890), (6, 1512))
    matrix_power_of_2_sizes = (
    (2, 2), (2, 4), (4, 2), (2, 8), (8, 2), (4, 8), (8, 4), (2, 512), (128, 128), (32, 1024), (1024, 256), (512, 8192))

    # timer
    start = time.perf_counter()

    print("Test that 'naiveFourierTransform' correctly transforms arrays of any size:")
    tests, total_tests, failures, total_failures = 0, 0, 0, 0
    for i, size in enumerate(array_any_sizes):
        if size > naive_size_thresh:
            break
        failures += executeAndReport(naiveFourierTransform_TransformAnyArray_CloseToActual, randomFloatArray(size))
        tests += 1
    printSummary(tests, failures)
    total_tests += tests
    total_failures += failures

    print("Test that 'naiveFourierTransform' correctly inverse transforms arrays of any size:")
    tests, failures = 0, 0
    for i, size in enumerate(array_any_sizes):
        if size > naive_size_thresh:
            break
        failures += executeAndReport(naiveFourierTransform_InverseTransformAnyArray_CloseToActual,
                                     randomTransformArray(size))
        tests += 1
    printSummary(tests, failures)
    total_tests += tests
    total_failures += failures

    print("Test that 'fastFourierTransform' correctly transforms arrays of size that are powers of 2:")
    tests, failures = 0, 0
    for i, size in enumerate(array_power_of_2_sizes):
        if size > fast_size_thresh:
            break
        failures += executeAndReport(fastFourierTransform_TransformPowerOf2SizedArray_CloseToActual,
                                     randomFloatArray(size))
        tests += 1
    printSummary(tests, failures)
    total_tests += tests
    total_failures += failures

    print("Test that 'fastFourierTransform' rejects to transform arrays of size that are not powers of 2:")
    tests, failures = 0, 0
    for i, size in enumerate(array_any_sizes):
        if size > fast_size_thresh:
            break
        if fft.isPowerOf2(size):
            continue
        failures += executeAndReport(fastFourierTransform_TransformNonPowerOf2SizedArray_ErrorRaised,
                                     randomFloatArray(size))
        tests += 1
    printSummary(tests, failures)
    total_tests += tests
    total_failures += failures

    print("Test that 'fastFourierTransform' correctly inverse transforms arrays of size that are powers of 2:")
    tests, failures = 0, 0
    for i, size in enumerate(array_power_of_2_sizes):
        if size > fast_size_thresh:
            break
        failures += executeAndReport(fastFourierTransform_InverseTransformPowerOf2SizedArray_CloseToActual,
                                     randomTransformArray(size))
        tests += 1
    printSummary(tests, failures)
    total_tests += tests
    total_failures += failures

    print("Test that 'fastFourierTransform' rejects to inverse transform arrays of size that are not powers of 2:")
    tests, failures = 0, 0
    for i, size in enumerate(array_any_sizes):
        if size > fast_size_thresh:
            break
        if fft.isPowerOf2(size):
            continue
        failures += executeAndReport(fastFourierTransform_InverseTransformNonPowerOf2SizedArray_ErrorRaised,
                                     randomTransformArray(size))
        tests += 1
    printSummary(tests, failures)
    total_tests += tests
    total_failures += failures

    print("Test that 'naiveFourierTransformMatrix' correctly transforms matrices of any size:")
    tests, failures = 0, 0
    for i, size in enumerate(matrix_any_sizes):
        if size[0] * size[1] > naive_size_thresh:
            break
        failures += executeAndReport(naiveFourierTransformMatrix_TransformAnyMatrix_CloseToActual,
                                     randomFloatMatrix(size))
        tests += 1
    printSummary(tests, failures)
    total_tests += tests
    total_failures += failures

    print("Test that 'naiveFourierTransformMatrix' correctly inverse transforms matrices of any size:")
    tests, failures = 0, 0
    for i, size in enumerate(matrix_any_sizes):
        if size[0] * size[1] > naive_size_thresh:
            break
        failures += executeAndReport(naiveFourierTransformMatrix_InverseTransformAnyMatrix_CloseToActual,
                                     randomTransformMatrix(size))
        tests += 1
    printSummary(tests, failures)
    total_tests += tests
    total_failures += failures

    print("Test that 'fastFourierTransformMatrix' correctly transforms matrices of size that are powers of 2:")
    tests, failures = 0, 0
    for i, size in enumerate(matrix_power_of_2_sizes):
        if size[0] * size[1] > fast_size_thresh:
            break
        failures += executeAndReport(fastFourierTransformMatrix_TransformPowerOf2SizedMatrix_CloseToActual,
                                     randomFloatMatrix(size))
        tests += 1
    printSummary(tests, failures)
    total_tests += tests
    total_failures += failures

    print("Test that 'fastFourierTransformMatrix' rejects to transform matrices of size that are not powers of 2:")
    tests, failures = 0, 0
    for i, size in enumerate(matrix_any_sizes):
        if size[0] * size[1] > fast_size_thresh:
            break
        if fft.isPowerOf2(size[0]) or fft.isPowerOf2(size[1]):
            continue
        failures += executeAndReport(fastFourierTransform_TransformNonPowerOf2SizedArray_ErrorRaised,
                                     randomFloatMatrix(size))
        tests += 1
    printSummary(tests, failures)
    total_tests += tests
    total_failures += failures

    print("Test that 'fastFourierTransformMatrix' correctly inverse transforms matrices of size that are powers of 2:")
    tests, failures = 0, 0
    for i, size in enumerate(matrix_power_of_2_sizes):
        if size[0] * size[1] > fast_size_thresh:
            break
        failures += executeAndReport(fastFourierTransformMatrix_InverseTransformPowerOf2SizedMatrix_CloseToActual,
                                     randomTransformMatrix(size))
        tests += 1
    printSummary(tests, failures)
    total_tests += tests
    total_failures += failures

    print(
        "Test that 'fastFourierTransformMatrix' rejects to inverse transform matrices of size that are not powers of 2:")
    tests, failures = 0, 0
    for i, size in enumerate(matrix_any_sizes):
        if size[0] * size[1] > fast_size_thresh:
            break
        if fft.isPowerOf2(size[0]) or fft.isPowerOf2(size[1]):
            continue
        failures += executeAndReport(fastFourierTransform_InverseTransformNonPowerOf2SizedArray_ErrorRaised,
                                     randomTransformMatrix(size))
        tests += 1
    printSummary(tests, failures)
    total_tests += tests
    total_failures += failures

    end = time.perf_counter()
    print("Finished test suite in {}s".format(end - start))
    printSummary(total_tests, total_failures)
    return total_failures


def main():
    total_failures = driver()
    if total_failures > 0:
        print("Test suite failed ({} failures)".format(total_failures))
    return total_failures


if __name__ == "__main__":
    return main()
