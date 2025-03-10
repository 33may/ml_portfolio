import numpy as np

from AIA.cnn.gradient_check import check_layer_gradient, check_layer_param_gradient
from AIA.cnn.layers import ConvolutionalLayerGPU, im2col, im2col_idx


def col2im(dx_patches, filter_size, stride):
    """

    Params:


    Returns:

    """
    # dx_patches [batch_size * out_height * out_width, filter_size * filter_size * in_channels]




# a = np.arange(16).reshape(1, 4, 4, 1)
#
# col = im2col(a, 3, 1)
#
# idx = im2col_idx(a, 3, 1)
#
# print("col")
# print(col)
#
# im = col2im(col, 3, 1)
#
# print("im")
# print(im)
#
# assert im == a


# X = np.array([
#     [
#         [[1.0, 0.0], [2.0, 1.0]],
#         [[0.0, -1.0], [-1.0, -2.0]]
#     ]
#     ,
#     [
#         [[0.0, 1.0], [1.0, -1.0]],
#         [[-2.0, 2.0], [-1.0, 0.0]]
#     ]
# ])
#
#
# layer = ConvolutionalLayerGPU(in_channels=2, out_channels=2, filter_size=3, padding=1)
# result = layer.forward(X)
# # Note this kind of layer produces the same dimensions as input
# assert result.shape == X.shape, "Result shape: %s - Expected shape %s" % (result.shape, X.shape)
# d_input = layer.backward(np.ones_like(result))
# # assert d_input.shape == X.shape
# layer = ConvolutionalLayerGPU(in_channels=2, out_channels=2, filter_size=3, padding=1)
# assert check_layer_param_gradient(layer, X, 'W')
# assert check_layer_param_gradient(layer, X, 'B')
# assert check_layer_gradient(layer, X)


# def im2col_idx_test(X_shape, filter_size, stride):
#     h, w = X_shape
#
#     y_start = np.arange(0, h - filter_size + 1, stride)
#     x_start = np.arange(0, w - filter_size + 1, stride)
#
#     # grid of initial coordinates
#     y_starts, x_starts = np.meshgrid(x_start, y_start, indexing='ij')  # [H_out, W_out]
#
#     dy, dx = np.mgrid[0:filter_size, 0:filter_size]  # [filter_size, filter_size]
#
#     y_indices = y_starts[:, :, None, None] + dy  # [H_out, W_out, filter_size, filter_size]
#     x_indices = x_starts[:, :, None, None] + dx  # [H_out, W_out, filter_size, filter_size]
#
#     return y_indices, x_indices
#
# def im2col_test(X, filter_size, stride):
#     h, w = X.shape
#
#     h_out = (h - filter_size) // stride + 1
#     w_out = (w - filter_size) // stride + 1
#
#     y_indices, x_indices= im2col_idx_test(X.shape, filter_size, stride)
#
#     patches = X[
#         y_indices,
#         x_indices,
#     ]  # [N, H_out, W_out, k, k, C]
#
#     result = patches.reshape(h_out * w_out, -1)
#
#     return result
#
#
# def col2im_test(col, x_shape, filter_size, stride):
#     h_out, w = col.shape
#     h_in, w_in = x_shape
#
#     result = np.zeros((h_in, w_in))
#
#     y_idx, x_idx = im2col_idx_test(x_shape, filter_size, stride)
#
#     y_idx_flat = y_idx.flatten()
#     x_indices_flat = x_idx.flatten()
#
#     col_flat = col.flatten()
#
#     np.add.at(result, (y_idx_flat, x_indices_flat), col_flat)
#
#     return result


def col2im(dx_patches, x_shape, filter_size, stride):
    batch_size, h_in, w_in, in_ch = x_shape

    result = np.zeros((batch_size, h_in, w_in, in_ch))

    batch_idx, y_idx, x_idx, chanel_idx = im2col_idx(x_shape, filter_size, stride)

    y_indices = y_idx[np.newaxis, :, :, :, :, np.newaxis]
    x_indices = x_idx[np.newaxis, :, :, :, :, np.newaxis]

    batch_idx = batch_idx + np.zeros_like(y_indices) + np.zeros_like(chanel_idx)
    chanel_idx = chanel_idx + np.zeros_like(y_indices) + np.zeros_like(batch_idx)
    y_idx = y_indices + np.zeros_like(chanel_idx) + np.zeros_like(batch_idx)
    x_idx = x_indices + np.zeros_like(chanel_idx) + np.zeros_like(batch_idx)

    batch_idx_flat = batch_idx.ravel()
    chanel_idx_flat = chanel_idx.ravel()
    y_idx_flat = y_idx.ravel()
    x_idx_flat = x_idx.ravel()

    dx_patches_flat = dx_patches.ravel()

    np.add.at(result, (batch_idx_flat, y_idx_flat, x_idx_flat, chanel_idx_flat), dx_patches_flat)

    return result


a = np.arange(9).reshape((1, 3, 3, 1))

# a = np.ones((2,3,3,2))

# dout = np.arange(16).reshape(2,2,2,2)

dout = im2col(a, 2, 1)

im = col2im(dout, a.shape, 2,  1)

print(33)