import time

from data_generator import DataProducer
from forward_pass_data import get_weights, get_bias, get_gamma, get_beta, get_mean, get_variance
from forward_pass_layers import convolution_forward, relu_forward, pool_forward, pure_batch_norm, fc_forward, sigmoid

def forward_pass(file, channels):
    data_generator = DataProducer()
    X = data_generator.load_data_with_channels(file, channels)

    X = X.reshape(1, X.shape[0], X.shape[1], X.shape[2])

    # CONV1



    W1 = get_weights(0, (1, 10, channels, 8), channels)
    b1 = get_bias(0, (1, 1, 1, 8), channels)
    out_conv_1 = convolution_forward(X, W1, b1, stride=1, padding=0)
    out_conv_1 = relu_forward(out_conv_1)
    X2 = pool_forward(out_conv_1, 1, 2, 1, 2)


    # CONV2
    W2 = get_weights(1, (1, 10, 8, 16), channels)
    b2 = get_bias(1, (1, 1, 1, 16), channels)
    out_conv_2 = convolution_forward(X2, W2, b2, stride=1, padding=0)
    out_conv_2 = relu_forward(out_conv_2)
    gamma1 = get_gamma(0, (16,), channels)
    beta1 = get_beta(0, (16,), channels)
    mean1 = get_mean(0, (16,), channels)
    variance1 = get_variance(0, (16,), channels)
    out_conv_2 = pure_batch_norm(out_conv_2, gamma1, beta1, mean1, variance1, 0.001)
    X3 = pool_forward(out_conv_2, 1, 2, 1, 2)


    # CONV3
    W3 = get_weights(2, (1, 10, 16, 32), channels)
    b3 = get_bias(2, (1, 1, 1, 32), channels)
    out_conv_3 = convolution_forward(X3, W3, b3, stride=1, padding=0)
    out_conv_3 = relu_forward(out_conv_3)
    gamma2 = get_gamma(1, (32,), channels)
    beta2 = get_beta(1, (32,), channels)
    mean2 = get_mean(1, (32,), channels)
    variance2 = get_variance(1, (32,), channels)
    out_conv_3 = pure_batch_norm(out_conv_3, gamma2, beta2, mean2, variance2, 0.001)
    X4 = pool_forward(out_conv_3, 1, 2, 1, 2)


    # CONV4
    W4 = get_weights(3, (1, 10, 32, 64), channels)
    b4 = get_bias(3, (1, 1, 1, 64), channels)
    out_conv_4 = convolution_forward(X4, W4, b4, stride=1, padding=0)
    out_conv_4 = relu_forward(out_conv_4)
    gamma3 = get_gamma(2, (64,), channels)
    beta3 = get_beta(2, (64,), channels)
    mean3 = get_mean(2, (64,), channels)
    variance3 = get_variance(2, (64,), channels)
    out_conv_4 = pure_batch_norm(out_conv_4, gamma3, beta3, mean3, variance3, 0.001)
    X5 = pool_forward(out_conv_4, 1, 2, 1, 2)

    X5 = X5.ravel().reshape(X.shape[0], -1)


    # FC1
    W5 = get_weights(4, (3456, 50), channels)
    b5 = get_bias(4, (50,), channels)
    out_fc_1 = fc_forward(X5, W5, b5)
    out_fc_1 = relu_forward(out_fc_1)
    gamma4 = get_gamma(3, (50,), channels)
    beta4 = get_beta(3, (50,), channels)
    mean4 = get_mean(3, (50,), channels)
    variance4 = get_variance(3, (50,), channels)
    out_fc_1 = pure_batch_norm(out_fc_1, gamma4, beta4, mean4, variance4, 0.001)
    start = time.clock()

    # FC2
    X6 = out_fc_1
    W6 = get_weights(5, (50, 1), channels)
    b6 = get_bias(5, (1,), channels)
    out_fc_2 = fc_forward(X6, W6, b6)
    out_fc_2 = sigmoid(out_fc_2)
    Y = 1 if out_fc_2 > 0.5 else 0
    return Y
