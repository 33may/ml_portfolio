import numpy as np


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
                res = patch_flat @ weights

                # add bias to result
                res += self.B.value

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

    batch_indices = np.arange(batch_size)[:, None, None, None, None]  # [N, 1, 1, 1, 1]
    channel_indices = np.arange(ch)[None, None, None, None, :]  # [1, 1, 1, 1, C]

    patches = X[
        batch_indices,
        y_indices,
        x_indices,
        channel_indices
    ]  # [N, H_out, W_out, k, k, C]

    if patches.ndim <= 5:
        if ch == 1:
            patches = patches[..., None]
        if batch_size == 1:
            patches = patches[..., None]

    im2col_matrix = patches.reshape(batch_size * h_out * w_out, filter_size * filter_size * ch)

    return im2col_matrix


class ConvolutionalLayerGPU:
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

        input_matrix = im2col(X, self.filter_size, 1) # [batch_size * height_out * width_out, filter_size * filter_size * in_channels]

        weights = self.W.value.reshape(self.in_channels * self.filter_size * self.filter_size, self.out_channels) # [in_channels * filter_size * filter_size, out_channels}

        result = input_matrix @ weights + self.B.value # [batch_size * height_out * width_out, out_channels]

        return result.reshape(batch_size, out_height, out_width, self.out_channels)



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

class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X = X

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        result = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                patch = X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                patch_max = np.max(patch, axis=(1, 2))

                result[:, y, x, :] = patch_max

        return result



    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape

        d_input = np.zeros_like(self.X)

        out_width = (width - self.pool_size) // self.stride + 1
        out_height = (height - self.pool_size) // self.stride + 1

        for y in range(out_height):
            for x in range(out_width):
                input_patch = self.X[:, y * self.stride:y * self.stride +self.pool_size, x * self.stride:x * self.stride +self.pool_size, :]

                input_patch_reshaped = input_patch.reshape(batch_size, -1, channels)

                max_idx_local = np.argmax(input_patch_reshaped, axis=1)

                row_idx_local, col_idx_local = np.unravel_index(max_idx_local, (self.pool_size, self.pool_size))

                row_idx_global = row_idx_local + y * self.stride
                col_idx_global = col_idx_local + x * self.stride

                batch_idx = np.arange(batch_size)[:, None]
                ch_idx = np.arange(channels)

                d_input[batch_idx, row_idx_global, col_idx_global, ch_idx] += d_out[:, y, x, :]

        return d_input


    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X_shape = X.shape

        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        return {}
