import numpy as np
def prod_non_zero_diag(matrix):
    return np.prod(np.diag(matrix)[np.nonzero(np.diag(matrix))])

def are_multisets_equal(left, right):
    return np.array_equal(np.sort(left), np.sort(right))

def max_after_zero(numbers):
    return np.max(np.extract((np.roll(numbers, 1) == 0)[1:], numbers[1:]))

def convert_image(image, channels):
    reshaped_channels = np.reshape(channels, (1, np.shape(channels)[0], 1))
    result = image.dot(np.reshape(channels, (1, np.shape(channels)[0], 1)))
    return np.reshape(result, (len(result), len(result[0])))

def run_length_encoding(numbers):
    starts = numbers != np.roll(numbers, 1)
    starts[0] = True
    fin = numbers != np.roll(numbers, -1)
    fin[-1] = True
    return (np.extract(starts, numbers), np.subtract(np.where(fin == True)[0] + 1, np.where(starts == True)[0]))

def pairwise_distance(left, right):
    left_n, left_m = np.shape(left)
    right_n, right_m = np.shape(right)
    return np.reshape(
            np.linalg.norm(np.reshape(np.tile(right, (left_n, 1)), (left_n * right_n, right_m)) -
            np.reshape(np.tile(left, right_n), (left_n * right_n, right_m)), axis = 1),
            (left_n, right_n)
    )

#print("prod_non_zero_diag result:", prod_non_zero_diag(np.array([[0, 2, 2], [1, 2, 4], [3, 4, 5]])))
#print("are_multisets_equal result:", are_multisets_equal(np.array([1, 2, 2, 4]),  np.array([4, 2, 1, 2])))
#print("max_after_zero result:", max_after_zero(np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])))
#print("convert_image result:", convert_image(
#    np.array([
#        np.array([np.array([1, 1]), np.array([2, 2])]),
#        np.array([np.array([3, 4]), np.array([2, 1])])
#    ]),
#    np.array([0.1, 0.1])))
#print("run_length_encoding result:", run_length_encoding(np.array([2])))
#print("pairwise_distance result:", pairwise_distance(np.array([[2, 2], [1, 3]]), np.array([[1, 1], [2, 2], [3, 3]])))


