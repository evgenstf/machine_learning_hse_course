import numpy as np
def prod_non_zero_diag(matrix):
    result = 1
    for i in range(len(matrix)):
        if (i < len(matrix[i])) and (matrix[i][i] != 0):
            result *= matrix[i][i]
    return result

def are_multisets_equal(left, right):
    if (len(left) != len(right)):
        return False
    left.sort()
    right.sort()
    for i in range(len(left)):
        if (left[i] != right[i]):
            return False
    return True

def max_after_zero(numbers):
    result = 0
    for i in range(1, len(numbers)):
        if (numbers[i - 1] == 0):
            result = max(result, numbers[i])
    return result

def convert_image(image, channels):
    new_image = []
    for row in image:
        new_row = []
        for pixel in row:
            new_pixel = 0
            for i in range(len(channels)):
                new_pixel += channels[i] * pixel[i]
            new_row.append(new_pixel)
        new_image.append(new_row)
    return new_image

def run_length_encoding(numbers):
    result = ([numbers[0]], [1])
    for i in range(1, len(numbers)):
        if (numbers[i] == result[0][-1]):
            result[1][-1] += 1
        else:
            result[0].append(numbers[i])
            result[1].append(1)
    return result;

def pairwise_distance(left, right):
    result = []
    for left_item in left:
        result.append([])
        for right_item in right:
            distance = 0
            for i in range(len(left_item)):
                distance += (left_item[i] - right_item[i]) ** 2
            distance = distance ** (1 / 2)
            result[-1].append(distance)
    return result

#print("are_multisets_equal result:", are_multisets_equal(np.array([1, 2, 2, 4]),  np.array([4, 2, 1, 2])))
#print("max_after_zero result:", max_after_zero(np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])))
#print("convert_image result:", convert_image([[[1, 1, 1]]], np.array([0.299, 0.587, 0.114])))
#print("run_length_encoding result:", run_length_encoding([2, 2, 2, 3, 4, 1, 1, 5]))
#print("pairwise_distance result:", pairwise_distance([[2, 2]], [[1, 1]]))
