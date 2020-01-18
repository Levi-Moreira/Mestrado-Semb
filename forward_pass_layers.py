import numpy as np

def relu_forward(X):
    out = np.maximum(X, 0)
    return out


def fc_forward(X, W, b):
    out = np.matmul(X, W) + b
    return out


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev, W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)
    ### END CODE HERE ###

    return Z


def convolution_forward(A_prev, W, b, stride, padding):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (f_h, f_w, _, n_C) = W.shape

    n_H = int((n_H_prev - f_h + 2 * padding) / stride) + 1
    n_W = int((n_W_prev - f_w + 2 * padding) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        a_prev_pad = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f_h
                    horiz_start = w * stride
                    horiz_end = horiz_start + f_w
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    conv_result = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])
                    Z[i, h, w, c] = conv_result

    assert (Z.shape == (m, n_H, n_W, n_C))

    return Z


def pool_forward(A_prev, f_h, f_w, stride_h, stride_w):
    """
    Implements the forward pass of the pooling layer
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """

    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f_h) / stride_h)
    n_W = int(1 + (n_W_prev - f_w) / stride_w)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):  # loop over the training examples
        for h in range(n_H):  # loop on the vertical axis of the output volume
            for w in range(0, n_W):  # loop on the horizontal axis of the output volume
                for c in range(0, n_C):  # loop over the channels of the output volume

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = stride_w * w
                    vert_end = vert_start + f_w

                    horiz_start = stride_h * h
                    horiz_end = horiz_start + f_h

                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[i, horiz_start:horiz_end, vert_start:vert_end, c]

                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.

                    if len(a_prev_slice.flatten()) > 0:
                        A[i, h, w, c] = np.max(a_prev_slice)

    # Making sure your output shape is correct
    assert (A.shape == (m, n_H, n_W, n_C))

    return A


def pure_batch_norm(X, gamma, beta, mean, variance, eps=1e-5):
    if len(X.shape) not in (2, 4):
        raise ValueError('only supports dense or 2dconv')

    # dense
    if len(X.shape) == 2:
        # normalize
        X_hat = (X - mean) * 1.0 / np.sqrt(variance + eps)
        # scale and shift
        out = gamma * X_hat + beta

    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, H, W, C = X.shape
        # normalize
        X_hat = (X - mean.reshape((1, 1, 1, C))) * 1.0 / np.sqrt(variance.reshape((1, 1, 1, C)) + eps)
        # scale and shift
        out = gamma.reshape((1, 1, 1, C)) * X_hat + beta.reshape((1, 1, 1, C))

    return out
