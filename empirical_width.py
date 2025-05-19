import numpy as np

def inverse_from_schur(inverse, row_add_to_kernel, schur):
    size = row_add_to_kernel.shape[1]
    A = np.zeros((size,size))
    B = row_add_to_kernel[0:1,:size-1]
    X = np.matmul(B,inverse)
    A_11 = inverse+np.matmul(X.T,X)/schur
    A[:size-1,:size-1] = A_11
    A[size-1,size-1] = 1/schur
    A[:size-1,(size-1):size] = -X.T/schur
    A[(size-1):size,:size-1] = -X/schur
    return A

def calc_sequence(sample, kernel_function, nSize=150):
    size = sample.shape[0]
    dim = sample.shape[1]
    sequence = []
    widths_squared = []
    diagonal_values = kernel_function(sample, sample, diagonal=1)
    x0 = np.argmax(diagonal_values)
    sequence.append(x0)
    widths_squared.append(diagonal_values[x0])
    current_sample = sample[sequence,:]
    inverse = np.array([[1/diagonal_values[x0]]])
    cur_vs_all = kernel_function(current_sample, sample, diagonal=0)
    for i in range(nSize):
        dot_products = np.matmul(inverse,cur_vs_all)
        values = np.sum(np.multiply(cur_vs_all,dot_products),axis=0)
        next_x = np.argmax(diagonal_values-values)
        sequence.append(next_x)
        if diagonal_values[next_x]-values[next_x]<1e-10:
            continue
        widths_squared.append(diagonal_values[next_x]-values[next_x])
        current_sample = sample[sequence,:]
        row_add_to_kernel = kernel_function(np.reshape(sample[next_x], (1,-1)), current_sample, diagonal=0)
        inverse = inverse_from_schur(inverse, row_add_to_kernel, diagonal_values[next_x]-values[next_x])
        row_add = kernel_function(np.reshape(sample[next_x], (1,-1)), sample, diagonal=0)
        cur_vs_all = np.concatenate((cur_vs_all, row_add), axis=0)
    return widths_squared

def kernel_delta_from_euclidean_on_sphere(euclead_delta, dim, kernel_function):
    one_hot = np.zeros((1,dim))
    second = np.zeros((1,dim))
    one_hot[0][0]=1.0
    alpha = (2-euclead_delta**2)/2
    second[0][0] = alpha
    second[0][1] = np.sqrt(1-alpha**2)
    kernel_delta = np.sqrt(kernel_function(one_hot,one_hot,diagonal=0)+                           kernel_function(second,second,diagonal=0)-                           2*kernel_function(one_hot,second,diagonal=0))
    return kernel_delta[0][0]
