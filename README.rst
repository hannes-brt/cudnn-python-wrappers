cudnn-python-wrappers
=====================

Python wrappers for the NVIDIA cuDNN libraries.
-----------------------------------------------

This is a set of minimal Python wrappers for the `NVIDIA
cuDNN <https://developer.nvidia.com/cuDNN>`__ library of convolutional
neural network primitives. NVIDIA cuDNN is available free of charge, but
requires an NVIDIA developer account to download. Users should follow
the cuDNN API documentation to use these wrappers, as they faithfully
replicate the cuDNN C API.

These wrappers expose the full cuDNN API as Python functions, but are
minimalistic in that they don't implement any higher order
functionality, such as operating directly on data structures like
PyCUDA ``GPUArray`` or cudamat ``CUDAMatrix``. Since the interface
faithfully replicates the C API, the user is responsible for
allocating and deallocating handles to all cuDNN data structures and
passing references to arrays as pointers. However, cuDNN status codes
are translated to Python exceptions. The most common application for
these wrappers will be to be used along `PyCUDA 
<http://mathema.tician.de/software/pycuda/>`__, but they will work
equally well with other frameworks such as `CUDAMat
<https://github.com/cudamat/cudamat>`__.

Users need to make sure that they pass all arguments as the correct data
type, that is ``ctypes.c_void_p`` for all handles and array pointers and
``ctypes.c_int`` for all integer arguments and enums. Here is an example
on how to perform forward convolution on a PyCUDA ``GPUArray``:

.. code:: python

    import pycuda.autoinit
    from pycuda import gpuarray
    import libcudnn, ctypes
    import numpy as np

    # Create a cuDNN context
    cudnn_context = libcudnn.cudnnCreate()

    # Set some options and tensor dimensions
    accumulate = libcudnn.cudnnAccumulateResults['CUDNN_RESULT_NO_ACCUMULATE']
    tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
    data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
    convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
    convolution_path = libcudnn.cudnnConvolutionPath['CUDNN_CONVOLUTION_FORWARD']

    n_input = 100
    filters_in = 10
    filters_out = 8
    height_in = 20
    width_in = 20
    height_filter = 5
    width_filter = 5
    pad_h = 4
    pad_w = 4
    vertical_stride = 1
    horizontal_stride = 1
    upscalex = 1
    upscaley = 1

    # Input tensor
    X = gpuarray.to_gpu(np.random.rand(n_input, filters_in, height_in, width_in)
        .astype(np.float32))

    # Filter tensor
    filters = gpuarray.to_gpu(np.random.rand(filters_out,
        filters_in, height_filter, width_filter).astype(np.float32))

    #Descriptor for input
    X_desc = libcudnn.cudnnCreateTensor4dDescriptor()
    libcudnn.cudnnSetTensor4dDescriptor(X_desc, tensor_format, data_type,
        n_input, filters_in, height_in, width_in)

    # Filter descriptor
    filters_desc = libcudnn.cudnnCreateFilterDescriptor()
    libcudnn.cudnnSetFilterDescriptor(filters_desc, data_type, filters_out,
        filters_in, height_filter, width_filter)

    # Convolution descriptor
    conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
    libcudnn.cudnnSetConvolutionDescriptor(conv_desc, X_desc, filters_desc,
        pad_h, pad_w, vertical_stride, horizontal_stride, upscalex, upscaley,
        convolution_mode)

    # Get output dimensions (first two values are n_input and filters_out)
    _, _, height_output, width_output = libcudnn.cudnnGetOutputTensor4dDim(
        conv_desc, convolution_path)

    # Output tensor
    Y = gpuarray.empty((n_input, filters_out, height_output, width_output), np.float32)
    Y_desc = libcudnn.cudnnCreateTensor4dDescriptor()
    libcudnn.cudnnSetTensor4dDescriptor(Y_desc, tensor_format, data_type, n_input,
        filters_out, height_output, width_output)

    # Get pointers to GPU memory
    X_data = ctypes.c_void_p(int(X.gpudata))
    filters_data = ctypes.c_void_p(int(filters.gpudata))
    Y_data = ctypes.c_void_p(int(Y.gpudata))

    # Perform convolution
    libcudnn.cudnnConvolutionForward(cudnn_context, X_desc, X_data,
        filters_desc, filters_data, conv_desc,
        Y_desc, Y_data, accumulate)

    # Clean up
    libcudnn.cudnnDestroyTensor4dDescriptor(X_desc)
    libcudnn.cudnnDestroyTensor4dDescriptor(Y_desc)
    libcudnn.cudnnDestroyFilterDescriptor(filters_desc)
    libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
    libcudnn.cudnnDestroy(cudnn_context)

Installation
------------

Install from PyPi with

::

    pip install cudnn-python-wrappers

