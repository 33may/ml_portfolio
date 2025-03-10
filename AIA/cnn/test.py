import numpy as np

def im2col(X, filter_size, stride):
    """
    Convert images into matrix suitable to apply GPU Convolution operations.

    Params:
        X: images batched with shape (batch_size, height, width, in_channels)
        filter_size: filter size
        stride: stride

    Returns:
        Matrix with shape (batch_size * height_out * width_out, filter_size * filter_size * in_channels)
    """
    # get the input tensor shape
    batch_size, h, w, ch = X.shape

    # compute the x,y coordinates of each possible patch combination start
    y_start = np.arange(0, h - filter_size + 1, stride)
    x_start = np.arange(0, w - filter_size + 1, stride)

    # compute the shape of the matrix after
    h_out = (h - filter_size) // stride + 1
    w_out = (w - filter_size) // stride + 1

    # grid of initial coordinates
    y_starts, x_starts  = np.meshgrid(x_start, y_start, indexing='ij') # [H_out, W_out]

    dy, dx = np.mgrid[0:filter_size, 0:filter_size] # [filter_size, filter_size]

    y_indices = y_starts[:, :, None, None] + dy  # [H_out, W_out, filter_size, filter_size]
    x_indices = x_starts[:, :, None, None] + dx  # [H_out, W_out, filter_size, filter_size]

    batch_indices = np.arange(batch_size)[:, None, None, None, None, None]  # [N, 1, 1, 1, 1]
    channel_indices = np.arange(ch)[None, None, None, None, None, :]  # [1, 1, 1, 1, C]

    patches = X[
        batch_indices,
        y_indices,
        x_indices,
        channel_indices
    ]  # [N, H_out, W_out, k, k, C]

    a = np.arange(16).reshape(4, 4)

    im2col_matrix = patches.reshape(batch_size * h_out * w_out, filter_size * filter_size * ch)

    return im2col_matrix


def col2im(dx_patches, filter_size, stride):
    """

    Params:


    Returns:

    """
    # dx_patches [batch_size * out_height * out_width, filter_size * filter_size * in_channels]



X = np.arange(16).reshape(1,4,4,1)

im2col(X, 3, 1)