from AIA.cnn.layers import ConvolutionalLayerGPU


def softmax(x):
    x_shifted = x - np.max(x, axis=0, keepdims=True)
    exps = np.exp(x_shifted)
    return exps / np.sum(exps, axis=0, keepdims=True)


def l2_regularization(W, reg_strength):
    loss = np.sum(np.square(W)) * reg_strength
    grad = 2 * reg_strength * W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    probs = softmax(predictions)

    loss = -np.sum(np.log(probs[target_index]))

    dprediction = probs[target_index]

    dprediction[target_index] -= 1

    return loss, dprediction


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
        self.W = Param(0.001 * cp.random.randn(n_input, n_output))
        self.B = Param(0.001 * cp.random.randn(1, n_output))
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
        return {'W': self.W, 'B': self.B}


class Model:
    def __init__(self, layers):
        """
        Initialize the model with a list of layers.
        """
        self.layers = layers

    def forward(self, x):
        """
        Perform a forward pass through all layers.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss_and_grad(self, predictions, target_index):
        """
        Compute the softmax with cross entropy loss and return both the loss
        and the gradient.
        """
        loss, grad = softmax_with_cross_entropy(predictions, target_index)
        return loss, grad

    def backward(self, grad):
        """
        Backpropagate the gradient through all layers.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self):
        """
        Collects parameters from all layers.
        Returns a dictionary mapping layer names to their parameters.
        """
        parameters = {}
        for idx, layer in enumerate(self.layers):
            for name, param in layer.params().items():
                parameters[f"layer{idx}_{name}"] = param
        return parameters

    def update_params(self, lr):
        """
        Updates all parameters.
        """
        for param in self.params().values():
            param.value -= lr * param.grad
            param.grad = np.zeros_like(param.grad)

    def train(self, lr, epochs, data_loader):
        """
        Trains the model.
        data_loader should yield batches as (x, y) where:
          - x is input with shape (batch_size, input_size)
          - y contains the target class indices for each sample.
        """
        loss_list = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in data_loader:
                # Forward pass: compute predictions for the current batch.
                predictions = self.forward(x)

                # Compute loss and gradient at the output.
                loss, grad = self.compute_loss_and_grad(predictions, y)
                epoch_loss += loss

                # Backward pass: propagate gradients through the network.
                self.backward(grad)

                # Update all parameters using gradient descent.
                self.update_params(lr)

            avg_loss = epoch_loss / len(data_loader)
            loss_list.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Plot training loss.
        plt.plot(loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()


import numpy as np
import cupy as cp


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    """
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    """
    # assert isinstance(x, np.ndarray)
    # assert x.dtype == float

    x = np.array(x.get() if isinstance(x, cp.ndarray) else x, copy=True, order='C')
    x.setflags(write=True)  # Explicitly make writeable

    fx, analytic_grad = f(x)
    analytic_grad = analytic_grad.copy()

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]

        arg_array_minus = np.copy(x)
        arg_array_plus = np.copy(x)
        arg_array_minus[ix] -= delta
        arg_array_plus[ix] += delta

        fx_plus, _ = f(arg_array_plus)
        fx_minus, _ = f(arg_array_minus)

        numeric_grad_at_ix = (fx_plus - fx_minus) / (2 * delta)

        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (
                ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True


def check_layer_gradient(layer, x, delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for the input and output of a layer

    Arguments:
      layer: neural network layer, with forward and backward functions
      x: starting point for layer input
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Returns:
      bool indicating whether gradients match or not
    """
    output = layer.forward(x)
    output_weight = np.random.randn(*output.shape)

    def helper_func(x):
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = np.ones_like(output) * output_weight
        grad = layer.backward(d_out)
        return loss, grad

    return check_gradient(helper_func, x, delta, tol)


def check_layer_param_gradient(layer, x,
                               param_name,
                               delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for the parameter of the layer

    Arguments:
      layer: neural network layer, with forward and backward functions
      x: starting point for layer input
      param_name: name of the parameter (W, B)
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Returns:
      bool indicating whether gradients match or not
    """
    param = layer.params()[param_name]
    initial_w = param.value

    output = layer.forward(x)
    output_weight = np.random.randn(*output.shape)

    def helper_func(w):
        param.value = cp.asarray(w) if isinstance(layer, ConvolutionalLayerGPU) else w
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = np.ones_like(output) * output_weight
        layer.backward(d_out)
        grad = cp.asnumpy(param.grad) if isinstance(param.grad, cp.ndarray) else param.grad
        return loss, grad

    return check_gradient(helper_func, initial_w, delta, tol)


def check_model_gradient(model, X, y,
                         delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for all model parameters

    Returns:
      bool indicating whether gradients match or not
    """
    params = model.params()

    for param_key in params:
        print("Checking gradient for %s" % param_key)
        param = params[param_key]
        initial_w = param.value

        def helper_func(w):
            param.value = w
            loss = model.compute_loss_and_gradients(X, y)
            grad = param.grad
            return loss, grad

        if not check_gradient(helper_func, initial_w, delta, tol):
            return False

    return True