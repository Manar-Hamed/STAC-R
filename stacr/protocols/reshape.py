import torch


class Reshaper:
    @staticmethod
    def to_tensor(matrix_input, target_shape):
        """
        Input: (Batch, N) e.g., (64, 512)
        Output: (Batch, d1, d2, d3) e.g., (64, 8, 8, 8)
        """
        B = matrix_input.shape[0]
        prod = 1
        for dim in target_shape:
            prod *= dim
        assert prod == matrix_input.shape[1], "Target shape must multiply to feature dimension"
        return matrix_input.reshape(B, *target_shape)

    @staticmethod
    def to_matrix(tensor_input):
        """
        Input: (Batch, d1, d2, d3)
        Output: (Batch, N)
        """
        B = tensor_input.shape[0]
        return tensor_input.reshape(B, -1)
