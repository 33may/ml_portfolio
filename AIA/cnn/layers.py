import numpy as np
import cupy as cp


class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = X > 0
        X = X * self.mask
        return X


    def backward(self, d_out):

        d_result = d_out * self.mask

        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X

        return X @ self.W.value + self.B.value

    def backward(self, d_out):

        d_input = d_out @ self.W.value.T

        d_W = self.X.T @ d_out
        self.W.grad += d_W

        d_B = np.sum(d_out, axis=0, keepdims=True)
        self.B.grad += d_B

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


def apply_padding(x, padding):
    batch_size, height, width, channels = x.shape

    template = np.zeros((batch_size, height + 2 * padding, width + 2 * padding, channels))

    template[:, padding:-padding, padding:-padding, :] = x

    return template

class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size, in_channels, out_channels) * np.sqrt(2.0 / (filter_size * filter_size * in_channels))
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

        self.last_X = None


    def forward(self, X):
        # get shape of the input tensor
        batch_size, height, width, in_channels = X.shape


        # compute shape of output tensor
        out_height = height - self.filter_size + 1 + 2 * self.padding
        out_width = width - self.filter_size + 1 + 2 * self.padding

        # pad the input tensor
        X = apply_padding(X, self.padding) if self.padding > 0 else X

        # save last input for backward pass
        self.last_X = X

        # create zeros tensor for result
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))

        # reshape weights to use matrix multiplication trick
        weights = self.W.value.reshape(self.in_channels * self.filter_size * self.filter_size, self.out_channels)

        # iterate each pixel in output tensor
        for y in range(out_height):
            for x in range(out_width):
                # take the perception widow of the output pixel
                patch = X[:, y:y + self.filter_size, x:x + self.filter_size, :]

                # unwrap patch to use matrix multiplication trick
                patch_flat = patch.reshape(batch_size, self.in_channels * self.filter_size * self.filter_size)

                # convolution operation
                res = patch_flat @ weights + self.B.value

                # add pixels to result tensor
                result[:, y, x, :] = res

        return result


    def backward(self, d_out):
        batch_size, height, width, in_channels = self.last_X.shape

        _, out_height, out_width, out_channels = d_out.shape

        result = np.zeros_like(self.last_X)

        weights = self.W.value.reshape(self.in_channels * self.filter_size * self.filter_size, self.out_channels)

        for y in range(out_height):
            for x in range(out_width):
                # take the gradient patch (batch_size, out_channels)
                gradient_patch = d_out[:, y, x, :]

                input_patch = self.last_X[:, y:y + self.filter_size, x:x + self.filter_size, :]

                input_patch_flat = input_patch.reshape(batch_size, self.in_channels * self.filter_size * self.filter_size)


                # d_x -> d_out + weights

                d_input_flat = gradient_patch @ weights.T

                d_input_patch = d_input_flat.reshape(batch_size, self.filter_size, self.filter_size, in_channels)

                result[:, y:y+self.filter_size, x:x+self.filter_size, :] += d_input_patch


                # d_w -> d_out + inputs
                d_flat_w = input_patch_flat.T @ gradient_patch

                d_w = d_flat_w.reshape(self.filter_size, self.filter_size, self.in_channels, self.out_channels)

                self.W.grad += d_w


                # d_b -> d_out

                d_b = np.sum(gradient_patch, axis=0)

                self.B.grad += d_b


        return result[:, self.padding:-self.padding, self.padding:-self.padding, :] if self.padding > 0 else result


    def params(self):
        return { 'W': self.W, 'B': self.B }



def im2col_idx(x_shape, filter_size, stride):
    batch_size, h, w, ch = x_shape

    y_start = np.arange(0, h - filter_size + 1, stride)
    x_start = np.arange(0, w - filter_size + 1, stride)

    # grid of initial coordinates
    y_starts, x_starts = np.meshgrid(x_start, y_start, indexing='ij')  # [H_out, W_out]

    dy, dx = np.mgrid[0:filter_size, 0:filter_size]  # [filter_size, filter_size]

    y_indices = y_starts[:, :, None, None] + dy  # [H_out, W_out, filter_size, filter_size]
    x_indices = x_starts[:, :, None, None] + dx  # [H_out, W_out, filter_size, filter_size]

    batch_indices = np.arange(batch_size)[:, None, None, None, None, None]  # [N, 1, 1, 1, 1]
    channel_indices = np.arange(ch)[None, None, None, None, None, :]  # [1, 1, 1, 1, C]

    return batch_indices, y_indices, x_indices, channel_indices


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

    h_out = (h - filter_size) // stride + 1
    w_out = (w - filter_size) // stride + 1

    batch_indices, y_indices, x_indices, channel_indices = im2col_idx(X.shape, filter_size, stride)

    y_indices = y_indices[np.newaxis, :, :, :, :, np.newaxis]
    x_indices = x_indices[np.newaxis, :, :, :, :, np.newaxis]

    patches = X[
        batch_indices,
        y_indices,
        x_indices,
        channel_indices
    ]  # [N, H_out, W_out, k, k, C]

    im2col_matrix = patches.reshape(batch_size * h_out * w_out, filter_size * filter_size * ch)

    return im2col_matrix


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

def im2col_idx(x_shape, filter_size, stride):
    """
    Generate indices required for extracting image patches for convolution operation.

    Params:
        x_shape: Tuple representing the shape of input tensor (batch_size, height, width, channels)
        filter_size: Size of convolutional filter (assumed square)
        stride: Stride size for sliding the filter

    Returns:
        batch_indices: Indices corresponding to batch dimension
        y_indices: Indices for each patch along height dimension
        x_indices: Indices for each patch along width dimension
        channel_indices: Indices corresponding to channel dimension
    """
    batch_size, ch, h, w = x_shape

    # y_start for height patch starts, x_start for width patch starts
    y_start = np.arange(0, h - filter_size + 1, stride)
    x_start = np.arange(0, w - filter_size + 1, stride)

    # meshgrid(y_start, x_start) for in-kernel positions
    y_starts, x_starts = np.meshgrid(y_start, x_start, indexing='ij')  # [H_out, W_out]

    dy, dx = np.mgrid[0:filter_size, 0:filter_size]  # [filter_size, filter_size]

    y_indices = y_starts[:, :, None, None] + dy  # [H_out, W_out, k, k]
    x_indices = x_starts[:, :, None, None] + dx  # [H_out, W_out, k, k]

    # Add batch and channel dimensions
    batch_indices = np.arange(batch_size)[:, None, None, None, None, None]  # [N, 1, 1, 1, 1, 1]
    channel_indices = np.arange(ch)[None, None, None, :, None, None]  # [1, 1, 1, C, 1, 1]
    # the order plays crucial role in proper ordering when indexing and reshaping

    return batch_indices, y_indices, x_indices, channel_indices


def im2col_matrix(X, filter_size, stride):
    """
    Converts batched images into a matrix format suitable for efficient GPU-based convolution operations.

    Params:
        X: Input tensor of images with shape (batch_size, height, width, in_channels)
        filter_size: Size of convolutional filter (assumed square)
        stride: Stride size for sliding the convolutional filter

    Returns:
        A reshaped matrix suitable for convolution with shape:
        (batch_size * height_out * width_out, filter_size * filter_size * in_channels)
    """
    batch_size, ch, h, w = X.shape
    h_out = (h - filter_size) // stride + 1
    w_out = (w - filter_size) // stride + 1

    batch_indices, y_indices, x_indices, channel_indices = im2col_idx(X.shape, filter_size, stride)

    # Expand indices for broadcasting
    y_indices_exp = y_indices[np.newaxis, :, :, np.newaxis, :, :]  # [1, 1, H_out, W_out, k, k]
    x_indices_exp = x_indices[np.newaxis, :, :, np.newaxis, :, :]  # [1, 1, H_out, W_out, k, k]

    # Extract patches
    patches = X[
        batch_indices,  # [N, 1, 1, 1, 1, 1]
        channel_indices,  # [1, C, 1, 1, 1, 1]
        y_indices_exp,  # [1, 1, H_out, W_out, k, k]
        x_indices_exp  # [1, 1, H_out, W_out, k, k]
    ]  # [N, C, H_out, W_out, k, k]

    # Reshape to [N * H_out * W_out, k * k * C]
    return patches.reshape(batch_size * h_out * w_out, filter_size * filter_size * ch)


def col2im_backward(dx_patches, x_shape, filter_size, stride):
    batch_size, in_ch, h_in, w_in = x_shape

    # Initialize the result tensor to accumulate gradients
    result = np.zeros((batch_size, in_ch, h_in, w_in))

    # Retrieve indices used previously in im2col to map patches back to input positions
    batch_idx, y_idx, x_idx, channel_idx = im2col_idx(x_shape, filter_size, stride)

    y_idx = y_idx[np.newaxis, :, :, np.newaxis, :, :]
    x_idx = x_idx[np.newaxis, :, :, np.newaxis, :, :]

    # Expand indices dimensions for broadcasting compatibility
    batch_idx = batch_idx + np.zeros_like(y_idx) + np.zeros_like(channel_idx)
    channel_idx = channel_idx + np.zeros_like(y_idx) + np.zeros_like(batch_idx)
    y_idx = y_idx + np.zeros_like(channel_idx) + np.zeros_like(batch_idx)
    x_idx = x_idx + np.zeros_like(channel_idx) + np.zeros_like(batch_idx)

    # Flatten indices and gradients to enable accumulation
    batch_idx_flat = batch_idx.ravel()
    y_idx_flat = y_idx.ravel()
    x_idx_flat = x_idx.ravel()
    channel_idx_flat = channel_idx.ravel()
    dx_patches_flat = dx_patches.ravel()

    # Accumulate gradients from dx_patches back into original input tensor positions
    np.add.at(result, (batch_idx_flat, channel_idx_flat, y_idx_flat, x_idx_flat), dx_patches_flat)

    return result


class ConvolutionalLayerGPU:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            cp.random.randn(filter_size, filter_size, in_channels, out_channels) * np.sqrt(
                2.0 / (filter_size * filter_size * in_channels))
        )
        self.W.grad = cp.asarray(self.W.grad)

        self.B = Param(cp.zeros(out_channels))
        self.B.grad = cp.asarray(self.B.grad)

        self.padding = padding

        self.last_X = None
        self.last_X_shape = None
        # Store im2col matrix for backward pass
        self.im2col_input = None

    def forward(self, X):
        # get shape of the input tensor
        batch_size, in_channels, height, width = X.shape

        # compute shape of output tensor
        out_height = height - self.filter_size + 1 + 2 * self.padding
        out_width = width - self.filter_size + 1 + 2 * self.padding

        # pad the input tensor
        X = np.pad(X, ((0,0), (0,0), (self.padding,)*2, (self.padding,)*2))

        # save last input for backward pass
        self.last_X = X
        self.last_X_shape = X.shape

        self.im2col_input = im2col_matrix(X, self.filter_size,1)  # [batch_size * height_out * width_out, filter_size * filter_size * in_channels]

        weights = self.W.value.reshape(self.in_channels * self.filter_size * self.filter_size,
                                       self.out_channels)  # [in_channels * filter_size * filter_size, out_channels}

        # im2col matrix to GPU
        self.im2col_input = cp.asarray(self.im2col_input)

        result = self.im2col_input @ weights + self.B.value  # [batch_size * height_out * width_out, out_channels]

        return cp.asnumpy(result.reshape(batch_size, self.out_channels, out_height, out_width))


    def backward(self, d_out):
        batch_size, in_channels, height, width = self.last_X.shape

        _, out_channels, out_height, out_width = d_out.shape

        weights = self.W.value.reshape(self.in_channels * self.filter_size * self.filter_size,
                                       self.out_channels)  # [in_channels * filter_size * filter_size, out_channels]

        reshaped_d_out = d_out.reshape(batch_size * out_height * out_width,
                                       self.out_channels)  # [batch_size * out_height * out_width, out_channels]

        # reshaped d_out to GPU
        reshaped_d_out = cp.asarray(reshaped_d_out)

        # d_x -> d_out + weights
        backprop_matrix = reshaped_d_out @ weights.T  # [batch_size * out_height * out_width, filter_size * filter_size * in_channels]

        d_x = col2im_backward(cp.asnumpy(backprop_matrix), self.last_X_shape, self.filter_size, 1)

        # d_w -> d_out + inputs
        d_w = self.im2col_input.T @ reshaped_d_out  # [in_channels * filter_size * filter_size, out_channels]

        self.W.grad += d_w.reshape(self.W.value.shape)

        # d_b -> d_out

        d_b = np.sum(reshaped_d_out, axis=0)

        self.B.grad += d_b

        return d_x[:, :, self.padding:-self.padding, self.padding:-self.padding] if self.padding > 0 else d_x

    def params(self):
        return {'W': self.W, 'B': self.B}