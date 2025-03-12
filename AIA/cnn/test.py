import numpy as np

from AIA.cnn.gradient_check import check_layer_gradient, check_layer_param_gradient
from AIA.cnn.layers import ConvolutionalLayerGPU, im2col, im2col_idx

    im2col_matrix = patches.reshape(batch_size * h_out * w_out, filter_size * filter_size * ch)

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

X = np.arange(16).reshape(1,4,4,1)

im2col(X, 3, 1)