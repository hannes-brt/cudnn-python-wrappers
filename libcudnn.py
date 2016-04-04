"""
Python interface to the NVIDIA cuDNN library
"""

import sys
import ctypes
import ctypes.util

if sys.platform in ('linux2', 'linux'):
    _libcudnn_libname_list = ['libcudnn.so', 'libcudnn.so.4', 'libcudnn.so.4.0.4']
elif sys.platform == 'darwin':
    _libcudnn_libname_list = ['libcudnn.dylib', 'libcudnn.4.dylib']
elif sys.platform == 'win32':
    _libcudnn_libname_list = ['cudnn64_4.dll']
else:
    raise RuntimeError('unsupported platform')

_libcudnn = None
for _libcudnn_libname in _libcudnn_libname_list:
    try:
        _libcudnn = ctypes.cdll.LoadLibrary(_libcudnn_libname)
    except OSError:
        pass
    else:
        break
if _libcudnn is None:
    raise OSError('cuDNN library not found')

# cuDNN error
_libcudnn.cudnnGetErrorString.restype = ctypes.c_char_p
_libcudnn.cudnnGetErrorString.argtypes = [ctypes.c_int]
class cudnnError(Exception):
    def __init__(self, status):
        self.status = status
    def __str__(self):
        error = _libcudnn.cudnnGetErrorString(self.status)
        return '%s' % (error)


# Data layout specification
# cudnnTensorFormat_t is an enumerated type used by
# cudnnSetTensor4dDescriptor() to create a tensor with a pre-defined layout.
cudnnTensorFormat = {
     'CUDNN_TENSOR_NCHW': 0, # This tensor format specifies that the data is laid
                             # out in the following order: image, features map,
                             # rows, columns. The strides are implicitly defined
                             # in such a way that the data are contiguous in
                             # memory with no padding between images, feature
                             # maps, rows, and columns; the columns are the
                             # inner dimension and the images are the outermost
                             # dimension.
     'CUDNN_TENSOR_NHWC': 1 # This tensor format specifies that the data is laid
                            # out in the following order: image, rows, columns,
                            # features maps. The strides are implicitly defined in
                            # such a way that the data are contiguous in memory
                            # with no padding between images, rows, columns,
                            # and features maps; the feature maps are the
                            # inner dimension and the images are the outermost
                            # dimension.
}

# Data type
# cudnnDataType_t is an enumerated type indicating the data type to which a tensor
# descriptor or filter descriptor refers.
cudnnDataType = {
    'CUDNN_DATA_FLOAT': 0,  # The data is 32-bit single-precision floating point
                            # ( float ).
    'CUDNN_DATA_DOUBLE': 1, # The data is 64-bit double-precision floating point
                            # ( double ).
    'CUDNN_DATA_HALF': 2    # The data is 16-bit half-precision floating point
                            # ( half ).
}

# cudnnAddMode_t is an enumerated type used by cudnnAddTensor() to specify how
# a bias tensor is added to an input/output tensor.
cudnnAddMode = {
   'CUDNN_ADD_IMAGE': 0,
   'CUDNN_ADD_SAME_HW': 0,  # In this mode, the bias tensor is defined as one
                            # image with one feature map. This image will be
                            # added to every feature map of every image of the
                            # input/output tensor.
   'CUDNN_ADD_FEATURE_MAP': 1,
   'CUDNN_ADD_SAME_CHW': 1, # In this mode, the bias tensor is defined as one
                            # image with multiple feature maps. This image
                            # will be added to every image of the input/output
                            # tensor.
   'CUDNN_ADD_SAME_C': 2,   # In this mode, the bias tensor is defined as one
                            # image with multiple feature maps of dimension
                            # 1x1; it can be seen as an vector of feature maps.
                            # Each feature map of the bias tensor will be added
                            # to the corresponding feature map of all height-by-
                            # width pixels of every image of the input/output
                            # tensor.
   'CUDNN_ADD_FULL_TENSOR': 3 # In this mode, the bias tensor has the same
                            # dimensions as the input/output tensor. It will be
                            # added point-wise to the input/output tensor.
}

# cudnnConvolutionMode_t is an enumerated type used by
# cudnnSetConvolutionDescriptor() to configure a convolution descriptor. The
# filter used for the convolution can be applied in two different ways, corresponding
# mathematically to a convolution or to a cross-correlation. (A cross-correlation is
# equivalent to a convolution with its filter rotated by 180 degrees.)
cudnnConvolutionMode = {
    'CUDNN_CONVOLUTION': 0, # In this mode, a convolution operation will be done
                            # when applying the filter to the images.
    'CUDNN_CROSS_CORRELATION': 1 # In this mode, a cross-correlation operation will
                            # be done when applying the filter to the images.
}

# cudnnConvolutionFwdPreference_t is an enumerated type used by
# cudnnGetConvolutionForwardAlgorithm() to help the choice of the algorithm used for the
# forward convolution.
cudnnConvolutionFwdPreference = {
    'CUDNN_CONVOLUTION_FWD_NO_WORKSPACE':  0, # In this configuration, the routine
                        # cudnnGetConvolutionForwardAlgorithm() is guaranteed to return
                        # an algorithm that does not require any extra workspace to be
                        # provided by the user.
    'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST': 1, # In this configuration, the routine
                        # cudnnGetConvolutionForwardAlgorithm() will return the fastest
                        # algorithm regardless how much workspace is needed to execute it.
    'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT': 2 # In this configuration, the routine
                        # cudnnGetConvolutionForwardAlgorithm() will return the fastest
                        # algorithm that fits within the memory limit that the user provided.
}

# cudnnConvolutionFwdAlgo_t is an enumerated type that exposes the different algorithm
# available to execute the forward convolution operation.
cudnnConvolutionFwdAlgo = {
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM': 0, # This algorithm expresses the convolution
                        # as a matrix product without actually explicitly forming the matrix
                        # that holds the input tensor data.
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM': 1, # This algorithm expresses the convolution
                        # as a matrix product without actually explicitly forming the matrix
                        # that holds the input tensor data, but still needs some memory
                        # workspace to precompute some indices in order to facilitate the
                        # implicit construction of the matrix that holds the input tensor data.
    'CUDNN_CONVOLUTION_FWD_ALGO_GEMM': 2, # This algorithm expresses the convolution as an
                        # explicit matrix product. A significant memory workspace is needed to
                        # store the matrix that holds the input tensor data.
    'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT': 3, # This algorithm expresses the convolution as a
                        # direct convolution (e.g without implicitly or explicitly doing a
                        # matrix multiplication).
    'CUDNN_CONVOLUTION_FWD_ALGO_FFT': 4,
    'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING': 5
}

cudnnConvolutionBwdDataPreference = {
    'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE': 0,
    'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST': 1,
    'CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT': 2
}

cudnnConvolutionBwdDataAlgo = {
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0': 0,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1': 1,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT': 2,
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING': 3
}

cudnnConvolutionBwdFilterPreference = {
    'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE' : 0,
    'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST' : 1,
    'CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT' : 2,
}

cudnnConvolutionBwdFilterAlgo = {
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0' : 0,
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1' : 1,
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT' : 2,
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3' : 3
}

# cudnnSoftmaxAlgorithm_t is used to select an implementation of the softmax
# function used in cudnnSoftmaxForward() and cudnnSoftmaxBackward().
cudnnSoftmaxAlgorithm = {
    'CUDNN_SOFTMAX_FAST': 0,    # This implementation applies the straightforward
                                # softmax operation.
    'CUDNN_SOFTMAX_ACCURATE': 1, # This implementation applies a scaling to the input
                                # to avoid any potential overflow.
    'CUDNN_SOFTMAX_LOG' : 2     # This implementation applied the Log
                                # softmax operation, scaling the input to avoid any potential
                                # overflow.
}

# cudnnSoftmaxMode_t is used to select over which data the cudnnSoftmaxForward()
# and cudnnSoftmaxBackward() are computing their results.
cudnnSoftmaxMode = {
    'CUDNN_SOFTMAX_MODE_INSTANCE': 0,   # The softmax operation is computed per image (N)
                                        # across the dimensions C,H,W.
    'CUDNN_SOFTMAX_MODE_CHANNEL': 1     # The softmax operation is computed per spatial
                                        # location (H,W) per image (N) across the dimension
                                        # C.
}

# cudnnPoolingMode_t is an enumerated type passed to
# cudnnSetPoolingDescriptor() to select the pooling method to be used by
# cudnnPoolingForward() and cudnnPoolingBackward() .
cudnnPoolingMode = {
    'CUDNN_POOLING_MAX': 0,     # The maximum value inside the pooling window will
                                # be used.
    'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING': 1,   # The values inside the
                                # pooling window will be averaged and this count
                                # includes padded values.
    'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING': 2    # The values inside the
                                #  pooling window will be averaged and this count
                                # does not include padded values.
}

# cudnnActivationMode_t is an enumerated type used to select the neuron activation
# function used in cudnnActivationForward() and cudnnActivationBackward() .
cudnnActivationMode = {
    'CUDNN_ACTIVATION_SIGMOID': 0,  # sigmoid function
    'CUDNN_ACTIVATION_RELU': 1,     # rectified linear function
    'CUDNN_ACTIVATION_TANH': 2      # hyperbolic tangent function
}

# cudnnNanPropagation_t is an enumerated type to specify the propogation of Nan
cudnnNanPropagation = {
    'CUDNN_NOT_PROPAGATE_NAN': 0,
    'CUDNN_PROPAGATE_NAN': 1
}


def cudnnCheckStatus(status):
    """
    Raise cuDNN exception

    Raise an exception corresponding to the specified cuDNN error code.

    Parameters
    ----------
    status : int
        cuDNN error code
    """

    if status != 0:
        raise cudnnError(status)

# Helper functions

_libcudnn.cudnnGetVersion.restype = ctypes.c_size_t
_libcudnn.cudnnGetVersion.argtypes = []
def cudnnGetVersion():
    """
    Get cuDNN Version.
    """
    return _libcudnn.cudnnGetVersion()

_libcudnn.cudnnCreate.restype = int
_libcudnn.cudnnCreate.argtypes = [ctypes.c_void_p]
def cudnnCreate():
    """
    Initialize cuDNN.

    Initializes cuDNN and returns a handle to the cuDNN context.

    Returns
    -------

    handle : cudnnHandle
        cuDNN context
    """

    handle = ctypes.c_void_p()
    status = _libcudnn.cudnnCreate(ctypes.byref(handle))
    cudnnCheckStatus(status)
    return handle.value

_libcudnn.cudnnDestroy.restype = int
_libcudnn.cudnnDestroy.argtypes = [ctypes.c_void_p]
def cudnnDestroy(handle):
    """
    Release cuDNN resources.

    Release hardware resources used by cuDNN.

    Parameters
    ----------
    handle : cudnnHandle
        cuDNN context.
    """

    status = _libcudnn.cudnnDestroy(ctypes.c_void_p(handle))
    cudnnCheckStatus(status)

_libcudnn.cudnnSetStream.restype = int
_libcudnn.cudnnSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def cudnnSetStream(handle, id):
    """
    Set current cuDNN library stream.

    Parameters
    ----------
    handle : cudnnHandle
        cuDNN context.
    id : cudaStream
        Stream Id.
    """

    status = _libcudnn.cudnnSetStream(handle, id)
    cudnnCheckStatus(status)

_libcudnn.cudnnGetStream.restype = int
_libcudnn.cudnnGetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def cudnnGetStream(handle):
    """
    Get current cuDNN library stream.

    Parameters
    ----------
    handle : int
        cuDNN context.

    Returns
    -------
    id : int
        Stream ID.
    """

    id = ctypes.c_void_p()
    status = _libcudnn.cudnnGetStream(handle, ctypes.byref(id))
    cudnnCheckStatus(status)
    return id.value

_libcudnn.cudnnCreateTensorDescriptor.restype = int
_libcudnn.cudnnCreateTensorDescriptor.argtypes = [ctypes.c_void_p]
def cudnnCreateTensorDescriptor():
    """
    Create a Tensor descriptor object.

    Allocates a cudnnTensorDescriptor_t structure and returns a pointer to it.

    Returns
    -------
    tensor_descriptor : int
        Tensor descriptor.
    """

    tensor = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateTensorDescriptor(ctypes.byref(tensor))
    cudnnCheckStatus(status)
    return tensor.value

_libcudnn.cudnnSetTensor4dDescriptor.restype = int
_libcudnn.cudnnSetTensor4dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int]
def cudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w):
    """
    Initialize a previously created Tensor 4D object.

    This function initializes a previously created Tensor4D descriptor object. The strides of
    the four dimensions are inferred from the format parameter and set in such a way that
    the data is contiguous in memory with no padding between dimensions.

    Parameters
    ----------
    tensorDesc : cudnnTensorDescriptor
        Handle to a previously created tensor descriptor.
    format : cudnnTensorFormat
        Type of format.
    dataType : cudnnDataType
        Data type.
    n : int
        Number of images.
    c : int
        Number of feature maps per image.
    h : int
        Height of each feature map.
    w : int
        Width of each feature map.
    """

    status = _libcudnn.cudnnSetTensor4dDescriptor(tensorDesc, format, dataType,
                                                  n, c, h, w)
    cudnnCheckStatus(status)

_libcudnn.cudnnSetTensor4dDescriptorEx.restype = int
_libcudnn.cudnnSetTensor4dDescriptorEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                   ctypes.c_int, ctypes.c_int, ]
def cudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride):
    """"
    Initialize a Tensor descriptor object with strides.

    This function initializes a previously created generic Tensor descriptor object into a
    4D tensor, similarly to cudnnSetTensor4dDescriptor but with the strides explicitly
    passed as parameters. This can be used to lay out the 4D tensor in any order or simply to
    define gaps between dimensions.

    Parameters
    ----------
    tensorDesc : cudnnTensorDescriptor_t
        Handle to a previously created tensor descriptor.
    dataType : cudnnDataType
        Data type.
    n : int
        Number of images.
    c : int
        Number of feature maps per image.
    h : int
        Height of each feature map.
    w : int
        Width of each feature map.
    nStride : int
        Stride between two consective images.
    cStride : int
        Stride between two consecutive feature maps.
    hStride : int
        Stride between two consecutive rows.
    wStride : int
        Stride between two consecutive columns.
    """

    status = _libcudnn.cudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w,
                                                    nStride, cStride, hStride, wStride)
    cudnnCheckStatus(status)

_libcudnn.cudnnGetTensor4dDescriptor.restype = int
_libcudnn.cudnnGetTensor4dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ]
def cudnnGetTensor4dDescriptor(tensorDesc):
    """"
    Get parameters of a Tensor descriptor object.

    This function queries the parameters of the previouly initialized Tensor4D descriptor
    object.

    Parameters
    ----------
    tensorDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.

    Returns
    -------
    dataType : cudnnDataType
        Data type.
    n : int
        Number of images.
    c : int
        Number of feature maps per image.
    h : int
        Height of each feature map.
    w : int
        Width of each feature map.
    nStride : int
        Stride between two consective images.
    cStride : int
        Stride between two consecutive feature maps.
    hStride : int
        Stride between two consecutive rows.
    wStride : int
        Stride between two consecutive columns.
    """

    dataType = ctypes.c_int()
    n = ctypes.c_int()
    c = ctypes.c_int()
    h = ctypes.c_int()
    w = ctypes.c_int()
    nStride = ctypes.c_int()
    cStride = ctypes.c_int()
    hStride = ctypes.c_int()
    wStride = ctypes.c_int()

    status = _libcudnn.cudnnGetTensor4dDescriptor(tensorDesc, ctypes.byref(dataType), ctypes.byref(n),
                                                  ctypes.byref(c), ctypes.byref(h), ctypes.byref(w),
                                                  ctypes.byref(nStride), ctypes.byref(cStride),
                                                  ctypes.byref(hStride), ctypes.byref(wStride))
    cudnnCheckStatus(status)

    return dataType.value, n.value, c.value, h.value, w.value, nStride.value, cStride.value, \
        hStride.value, wStride.value

_libcudnn.cudnnDestroyTensorDescriptor.restype = int
_libcudnn.cudnnDestroyTensorDescriptor.argtypes = [ctypes.c_void_p]
def cudnnDestroyTensorDescriptor(tensorDesc):
    """"
    Destroy a Tensor descriptor.

    This function destroys a previously created Tensor descriptor object.

    Parameters
    ----------
    tensorDesc : cudnnTensorDescriptor
        Previously allocated Tensor descriptor object.
    """

    status = _libcudnn.cudnnDestroyTensorDescriptor(tensorDesc)
    cudnnCheckStatus(status)

_libcudnn.cudnnTransformTensor.restype = int
_libcudnn.cudnnTransformTensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p]
def cudnnTransformTensor(handle, alpha, srcDesc, srcData, beta, destDesc, destData):
    """"
    Tensor layout conversion helper (dest = alpha * src + beta * dest).

    This function copies the scaled data from one tensor to another tensor with a different
    layout. Those descriptors need to have the same dimensions but not necessarily the
    same strides. The input and output tensors must not overlap in any way (i.e., tensors
    cannot be transformed in place). This function can be used to convert a tensor with an
    unsupported format to a supported one.

    Parameters
    ----------
    handle : cudnnHandle
        cuDNN context.
    alpha : float
        Scalar factor to be applied to every element of the input tensor before it is added
        to the output tensor.
    srcDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    srcData : void_p
        Pointer to data of the tensor described by srcDesc descriptor.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior to adding
        the result of the operation. Note that if beta is zero, the output is not read and can
        contain any uninitialized data (including Nan numbers).
    destDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    destData : void_p
        Pointer to data of the tensor described by destDesc descriptor.
    """

    dataType, _, _, _, _, _, _, _, _ = cudnnGetTensor4dDescriptor(destDesc)
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnTransformTensor(handle, alphaRef, srcDesc,
                                                srcData, betaRef,
                                                destDesc, destData)
    cudnnCheckStatus(status)

_libcudnn.cudnnAddTensor.restype = int
_libcudnn.cudnnAddTensor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p]
def cudnnAddTensor(handle, mode, alpha, biasDesc, biasData, beta, srcDestDesc, srcDestData):
    """"
    Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc.

    This function adds the scaled values of one tensor to another tensor. The mode parameter
    can be used to select different ways of performing the scaled addition. The amount
    of data described by the biasDesc descriptor must match exactly the amount of data
    needed to perform the addition.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a cuDNN context.
    mode : cudnnAddMode
        Addition mode that describes how the addition is performed
    alpha : float
        Scalar factor to be applied to every data element of the bias tensor before it is added
        to the output tensor.
    biasDesc : cudnnTensorDescriptor
        Handle to a previoulsy initialized tensor descriptor.
    biasData : void_p
        Pointer to data of the tensor described by biasDesc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior to adding
        the result of the operation. Note that if beta is zero, the output is not read and can
        contain any uninitialized data (including Nan numbers).
    srcDestDesc : cudnnTensorDescriptor
        Handle to a previoulsy initialized tensor descriptor.
    srcDestData : void_p
        Pointer to data of the tensor described by srcDestDesc.
    """

    dataType, _, _, _, _, _, _, _, _ = cudnnGetTensor4dDescriptor(srcDestDesc)
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnAddTensor(handle, mode, alphaRef, biasDesc,
                                        biasData, betaRef,
                                        srcDestDesc, srcDestData)
    cudnnCheckStatus(status)

_libcudnn.cudnnSetTensor.restype = int
_libcudnn.cudnnSetTensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p]
def cudnnSetTensor(handle, srcDesc, srcData, value):
    """"
    Set all data points of a tensor to a given value : srcDest = alpha.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    srcDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    srcData : void_p
        Pointer to data of the tensor described by srcDesc descriptor.
    value : float
        Value that all elements of the tensor will be set to.
    """

    dataType, _, _, _, _, _, _, _, _ = cudnnGetTensor4dDescriptor(srcDesc)
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))

    status = _libcudnn.cudnnSetTensor(handle, srcDesc, srcData, alphaRef)
    cudnnCheckStatus(status)

_libcudnn.cudnnScaleTensor.restype = int
_libcudnn.cudnnScaleTensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p]
def cudnnScaleTensor(handle, srcDesc, srcData, alpha):
    """"
    This function scales all the elements of a tensor by a give factor.

    Set all data points of a tensor to scaled value : srcDest = alpha * srcDest.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    srcDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    srcData : void_p
        Pointer to data of the tensor described by srcDesc descriptor.
    alpha : float
        Value that all elements of the tensor will be scaled with.
    """

    dataType, _, _, _, _, _, _, _, _ = cudnnGetTensor4dDescriptor(srcDesc)
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))

    status = _libcudnn.cudnnScaleTensor(handle, srcDesc, srcData, alphaRef)
    cudnnCheckStatus(status)

_libcudnn.cudnnCreateFilterDescriptor.restype = int
_libcudnn.cudnnCreateFilterDescriptor.argtypes = [ctypes.c_void_p]
def cudnnCreateFilterDescriptor():
    """"
    Create a filter descriptor.

    This function creates a filter descriptor object by allocating the memory needed
    to hold its opaque structure.

    Parameters
    ----------

    Returns
    -------
    wDesc : cudnnFilterDescriptor
        Handle to a newly allocated filter descriptor.
    """

    wDesc = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateFilterDescriptor(ctypes.byref(wDesc))
    cudnnCheckStatus(status)

    return wDesc.value

_libcudnn.cudnnSetFilter4dDescriptor.restype = int
_libcudnn.cudnnSetFilter4dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_int, ctypes.c_int, ctypes.c_int]
def cudnnSetFilter4dDescriptor(wDesc, dataType, k, c, h, w):
    """"
    Initialize a filter descriptor.

    This function initializes a previously created filter descriptor object into a 4D filter.
    Filters layout must be contiguous in memory.

    Parameters
    ----------
    wDesc : cudnnFilterDescriptor
        Handle to a previously created filter descriptor.
    dataType : cudnnDataType
        Data type.
    k : int
        Number of output feature maps.
    c : int
        Number of input feature maps.
    h : int
        Height of each filter.
    w : int
        Width of each filter.
    """

    status = _libcudnn.cudnnSetFilter4dDescriptor(wDesc, dataType, k, c, h, w)
    cudnnCheckStatus(status)

_libcudnn.cudnnGetFilter4dDescriptor.restype = int
_libcudnn.cudnnGetFilter4dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cudnnGetFilter4dDescriptor(wDesc):
    """"
    Get parameters of filter descriptor.

    This function queries the parameters of the previouly initialized filter descriptor object.

    Parameters
    ----------
    wDesc : cudnnFilterDescriptor
        Handle to a previously created filter descriptor.

    Returns
    -------
    dataType : cudnnDataType
        Data type.
    k : int
        Number of output feature maps.
    c : int
        Number of input feature maps.
    h : int
        Height of each filter.
    w : int
        Width of each filter.
    """

    dataType = ctypes.c_int()
    k = ctypes.c_int()
    c = ctypes.c_int()
    h = ctypes.c_int()
    w = ctypes.c_int()

    status = _libcudnn.cudnnGetFilter4dDescriptor(wDesc, ctypes.byref(dataType),
                                                ctypes.byref(k), ctypes.byref(c),
                                                ctypes.byref(h), ctypes.byref(w))
    cudnnCheckStatus(status)

    return dataType.value, k.value, c.value, h.value, w.value

_libcudnn.cudnnDestroyFilterDescriptor.restype = int
_libcudnn.cudnnDestroyFilterDescriptor.argtypes = [ctypes.c_void_p]
def cudnnDestroyFilterDescriptor(wDesc):
    """"
    Destroy filter descriptor.

    This function destroys a previously created Tensor4D descriptor object.

    Parameters
    ----------
    wDesc : cudnnFilterDescriptor
    """

    status = _libcudnn.cudnnDestroyFilterDescriptor(wDesc)
    cudnnCheckStatus(status)

_libcudnn.cudnnCreateConvolutionDescriptor.restype = int
_libcudnn.cudnnCreateConvolutionDescriptor.argtypes = [ctypes.c_void_p]
def cudnnCreateConvolutionDescriptor():
    """"
    Create a convolution descriptor.

    This function creates a convolution descriptor object by allocating the memory needed to
    hold its opaque structure.

    Returns
    -------
    convDesc : cudnnConvolutionDescriptor
        Handle to newly allocated convolution descriptor.
    """

    convDesc = ctypes.c_void_p()

    status = _libcudnn.cudnnCreateConvolutionDescriptor(ctypes.byref(convDesc))
    cudnnCheckStatus(status)

    return convDesc.value

_libcudnn.cudnnSetConvolution2dDescriptor.restype = int
_libcudnn.cudnnSetConvolution2dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                    ctypes.c_int, ctypes.c_int]
def cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode):
    """"
    Initialize a convolution descriptor.

    This function initializes a previously created convolution descriptor object into a 2D
    correlation. This function assumes that the tensor and filter descriptors corresponds
    to the formard convolution path and checks if their settings are valid. That same
    convolution descriptor can be reused in the backward path provided it corresponds to
    the same layer.

    Parameters
    ----------
    convDesc : cudnnConvolutionDescriptor
        Handle to a previously created convolution descriptor.
    pad_h : int
        zero-padding height: number of rows of zeros implicitly concatenated
        onto the top and onto the bottom of input images.
    pad_w : int
        zero-padding width: number of columns of zeros implicitly concatenated
        onto the left and onto the right of input images.
    u : int
        Vertical filter stride.
    v : int
        Horizontal filter stride.
    upscalex : int
        Upscale the input in x-direction.
    uscaley : int
        Upscale the input in y-direction.
    mode : cudnnConvolutionMode
        Select between CUDNN_CONVOLUTION or CUDNN_CROSS_CORRELATION.
    """

    status = _libcudnn.cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v,
                                                        upscalex, upscaley, mode)
    cudnnCheckStatus(status)

_libcudnn.cudnnGetConvolution2dDescriptor.restype = int
_libcudnn.cudnnGetConvolution2dDescriptor.argtypes = [ctypes.c_void_p]
def cudnnGetConvolution2dDescriptor(convDesc):
    """"
    Get a convolution descriptor.

    This function queries a previously initialized 2D convolution descriptor object.

    Parameters
    ----------
    convDesc : cudnnConvolutionDescriptor
        Handle to a previously created convolution descriptor.

    Returns
    -------
    pad_h : int
        zero-padding height: number of rows of zeros implicitly concatenated onto
        the top and onto the bottom of input images.
    pad_w : int
        zero-padding width: number of columns of zeros implicitly concatenated
        onto the left and onto the right of input images.
    u : int
        Vertical filter stride.
    v : int
        Horizontal filter stride.
    upscalex : int
        Upscale the input in x-direction.
    upscaley : int
        Upscale the input in y-direction.
    mode : cudnnConvolutionMode
        Either CUDNN_CONVOLUTION or CUDNN_CROSS_CORRELATION.
    """
    pad_h = ctypes.c_int()
    pad_w = ctypes.c_int()
    u = ctypes.c_int()
    v = ctypes.c_int()
    upscalex = ctypes.c_int()
    upscaley = ctypes.c_int()
    mode = ctypes.c_int()

    status = _libcudnn.cudnnGetConvolution2dDescriptor(convDesc, ctypes.byref(pad_h),
                                                    ctypes.byref(pad_w), ctypes.byref(u),
                                                    ctypes.byref(v), ctypes.byref(upscalex),
                                                    ctypes.byref(upscaley),
                                                    ctypes.byref(mode))

    cudnnCheckStatus(status)

    return pad_h.value, pad_w.value, u.value, v.value, upscalex.value, upscaley.value, mode.value

_libcudnn.cudnnGetConvolution2dForwardOutputDim.restype = int
_libcudnn.cudnnGetConvolution2dForwardOutputDim.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensorDesc, wDesc):
    """"
    Return the dimensions of the output tensor given a convolution descriptor.

    This function returns the dimensions of the resulting 4D tensor of a 2D
    convolution, given the convolution descriptor, the input tensor descriptor and
    the filter descriptor. This function can help to setup the output tensor and allocate
    the proper amount of memory prior to launching the actual convolution.

    Parameters
    ----------
    convDesc : cudnnConvolutionDescriptor
        Handle to a previously created convolution descriptor.
    inputTensorDesc: cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    wDesc: cudnnFilterDescriptor
        Handle to a previously initialized filter descriptor.

    Returns
    -------
    n : int
        Number of output images.
    c : int
        Number of output feature maps per image.
    h : int
        Height of each output feature map.
    w : int
        Width of each output feature map.
    """
    n = ctypes.c_int()
    c = ctypes.c_int()
    h = ctypes.c_int()
    w = ctypes.c_int()

    status = _libcudnn.cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensorDesc,
                                                 wDesc, ctypes.byref(n),
                                                 ctypes.byref(c), ctypes.byref(h),
                                                 ctypes.byref(w))
    cudnnCheckStatus(status)

    return n.value, c.value, h.value, w.value

_libcudnn.cudnnSetConvolutionNdDescriptor.restype = int
_libcudnn.cudnnSetConvolutionNdDescriptor.argtypes = [ctypes.c_void_p, # convDesc
                                                      ctypes.c_int, # arrayLength
                                                      ctypes.POINTER(ctypes.c_int), # padA[]
                                                      ctypes.POINTER(ctypes.c_int), # filterStrideA[]
                                                      ctypes.POINTER(ctypes.c_int), # upscaleA[]
                                                      ctypes.c_int, # mode
                                                      ctypes.c_int] # dataType
def cudnnSetConvolutionNdDescriptor(convDesc, padA, filterStrideA, upscaleA, mode, dataType):
    dim = len(padA)
    status = _libcudnn.cudnnSetConvolutionNdDescriptor(convDesc,
                                                       dim,
                                                       (ctypes.c_int * dim)(*padA),
                                                       (ctypes.c_int*dim)(*filterStrideA),
                                                       (ctypes.c_int*dim)(*upscaleA),
                                                       mode,
                                                       dataType)
    cudnnCheckStatus(status)



_libcudnn.cudnnDestroyConvolutionDescriptor.restype = int
_libcudnn.cudnnDestroyConvolutionDescriptor.argtypes = [ctypes.c_void_p]
def cudnnDestroyConvolutionDescriptor(convDesc):
    """"
    Destroy a convolution descriptor.

    This function destroys a previously created convolution descriptor object.

    Parameters
    ----------
    convDesc : int
        Previously created convolution descriptor.
    """

    status = _libcudnn.cudnnDestroyConvolutionDescriptor(convDesc)
    cudnnCheckStatus(status)

class cudnnConvolutionFwdAlgoPerf(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_int),
                ("status", ctypes.c_int),
                ("time", ctypes.c_float),
                ("memory", ctypes.c_size_t)]

    def __str__(self):
        return '(algo=%d, status=%d, time=%f, memory=%d)' % (self.algo,
                                                             self.status,
                                                             self.time,
                                                             self.memory)
    def __repr__(self):
        return self.__str__()

_libcudnn.cudnnFindConvolutionForwardAlgorithm.restype = int
_libcudnn.cudnnFindConvolutionForwardAlgorithm.argtypes = [ctypes.c_void_p, # handle
                                                           ctypes.c_void_p, # xDesc
                                                           ctypes.c_void_p, # wDesc
                                                           ctypes.c_void_p, # convDesc
                                                           ctypes.c_void_p, # yDesc
                                                           ctypes.c_int, # requestAlgoCount
                                                           ctypes.c_void_p, #returnedAlgoCount
                                                           ctypes.c_void_p] #perfResults
def cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount):
    perfResultsType = cudnnConvolutionFwdAlgoPerf * requestedAlgoCount
    perfResults = perfResultsType()
    returnedAlgoCount = ctypes.c_int()
    status = _libcudnn.cudnnFindConvolutionForwardAlgorithm(handle,
                                                            xDesc,
                                                            wDesc,
                                                            convDesc,
                                                            yDesc,
                                                            ctypes.c_int(requestedAlgoCount),
                                                            ctypes.byref(returnedAlgoCount),
                                                            ctypes.cast(perfResults, ctypes.POINTER(cudnnConvolutionFwdAlgoPerf)))
    cudnnCheckStatus(status)
    return perfResults[0:returnedAlgoCount.value]


_libcudnn.cudnnGetConvolutionForwardAlgorithm.restype = int
_libcudnn.cudnnGetConvolutionForwardAlgorithm.argtypes = [ctypes.c_void_p,
                                                          ctypes.c_void_p, ctypes.c_void_p,
                                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                                          ctypes.c_size_t, ctypes.c_void_p]
def cudnnGetConvolutionForwardAlgorithm(handle, srcDesc, wDesc,
                                        convDesc, destDesc, preference, memoryLimitInbytes):
    """"
    This function returns the best algorithm to choose for the forward convolution
    depending on the critera expressed in the cudnnConvolutionFwdPreference_t enumerant.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    srcDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    wDesc : cudnnFilterDescriptor
        Handle to a previously initialized filter descriptor.
    convDesc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    destDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    preference : cudnnConvolutionFwdPreference
        Enumerant to express the preference criteria in terms of memory
        requirement and speed.
    memoryLimitInbytes: size_t
        The maximum amount of GPU memory the user is willing to use as a workspace
        when preference is CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT.

    Returns
    -------
    algo: cudnnConvolutionFwdAlgo
        Enumerant that specifies which convolution algorithm should be used to
        compute the results according to the specified preference.
    """
    algo = ctypes.c_int()

    status = _libcudnn.cudnnGetConvolutionForwardAlgorithm(handle, srcDesc, wDesc,
                                               convDesc, destDesc, preference,
                                               ctypes.c_size_t(memoryLimitInbytes),
                                               ctypes.byref(algo))
    cudnnCheckStatus(status)

    return algo

_libcudnn.cudnnGetConvolutionForwardWorkspaceSize.restype = int
_libcudnn.cudnnGetConvolutionForwardWorkspaceSize.argtypes = [ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_void_p,
                                                              ctypes.c_int]
def cudnnGetConvolutionForwardWorkspaceSize(handle, srcDesc, wDesc,
                                            convDesc, destDesc, algo):
    """"
    This function returns the amount of GPU memory workspace the user needs
    to allocate to be able to call cudnnConvolutionForward with the specified algorithm.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    srcDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    wDesc : cudnnFilterDescriptor
        Handle to a previously initialized filter descriptor.
    convDesc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    destDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    algo : cudnnConvolutionFwdAlgo
        Enumerant that specifies the chosen convolution algorithm.

    Returns
    -------
    sizeInBytes: c_size_t
        Amount of GPU memory needed as workspace to be able to execute a
        forward convolution with the sepcified algo.
    """
    sizeInBytes = ctypes.c_size_t()

    status = _libcudnn.cudnnGetConvolutionForwardWorkspaceSize(handle, srcDesc, wDesc,
                                               convDesc, destDesc, algo,
                                               ctypes.byref(sizeInBytes))
    cudnnCheckStatus(status)

    return sizeInBytes

_libcudnn.cudnnConvolutionForward.restype = int
_libcudnn.cudnnConvolutionForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_int,
                                              ctypes.c_void_p, ctypes.c_size_t,
                                              ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p]
def cudnnConvolutionForward(handle, alpha, srcDesc, srcData, wDesc, w,
                            convDesc, algo, workspace, workSpaceSizeInBytes, beta,
                            destDesc, destData):
    """"
    Perform forward convolution. All of the form "output = alpha * Op(inputs) + beta * output".

    This function executes convolutions or cross-correlations over src using the specified
    filters, returning results in dest. Scaling factors alpha and beta can be used to scale
    the input tensor and the output tensor respectively.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    alpha: float
        Scaling factor with which every element of the input tensor is multiplied.
    srcDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor srcDesc.
    wDesc : cudnnFilterDescriptor
        Handle to a previously initialized filter descriptor.
    w : void_p
        Data pointer to GPU memory associated with the filter descriptor wDesc.
    convDesc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    algo: cudnnConvolutionFwdAlgo
        Enumerant that specifies which convolution algorithm shoud be used to
        compute the results.
    workSpace: void_p
        Data pointer to GPU memory to a workspace needed to able to execute
        the specified algorithm. If no workspace is needed for a particular
        algorithm, that pointer can be nil.
    workSpaceSizeInBytes: long
        Specifies the size in bytes of the provided workSpace.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the convolution.
    destDesc : cudnnTensorDescriptor
        Handle to a previously initialized tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the tensor descriptor destDesc.
    """

    dataType = cudnnGetTensor4dDescriptor(destDesc)[0]
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnConvolutionForward(handle, alphaRef, srcDesc, srcData,
                                            wDesc, w,
                                            convDesc, algo, workspace,
                                            ctypes.c_size_t(workSpaceSizeInBytes),
                                            betaRef, destDesc, destData)
    cudnnCheckStatus(status)

_libcudnn.cudnnConvolutionBackwardBias.restype = int
_libcudnn.cudnnConvolutionBackwardBias.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p]
def cudnnConvolutionBackwardBias(handle, alpha, srcDesc, srcData, beta, destDesc, destData):
    """"
    Compute the gradient wrt the bias.

    This function computes the convolution gradient with respect to the bias, which is the
    sum of every element belonging to the same feature map across all of the images of the
    input tensor. Therefore, the number of elements produced is equal to the number of
    features maps of the input tensor.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    alpha: float
        Scaling factor with which every element of the input tensor is multiplied.
    srcDesc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the convolution gradient. Note that if beta is zero,
        the output is not read and can contain any uninitialized data (including
        Nan numbers).
    destDesc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDesc.
    """

    dataType = cudnnGetTensor4dDescriptor(destDesc)[0]
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnConvolutionBackwardBias(handle, alphaRef, srcDesc, srcData,
                                                    betaRef, destDesc, destData)
    cudnnCheckStatus(status)


class cudnnConvolutionBwdDataAlgoPerf(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_int),
                ("status", ctypes.c_int),
                ("time", ctypes.c_float),
                ("memory", ctypes.c_size_t)]

    def __str__(self):
        return '(algo=%d, status=%d, time=%f, memory=%d)' % (self.algo,
                                                             self.status,
                                                             self.time,
                                                             self.memory)
    def __repr__(self):
        return self.__str__()

_libcudnn.cudnnFindConvolutionBackwardDataAlgorithm.restype = int
_libcudnn.cudnnFindConvolutionBackwardDataAlgorithm.argtypes = [ctypes.c_void_p, # handle
                                                                ctypes.c_void_p, # wDesc
                                                                ctypes.c_void_p, # dyDesc
                                                                ctypes.c_void_p, # convDesc
                                                                ctypes.c_void_p, # dxDesc
                                                                ctypes.c_int, # requestAlgoCount
                                                                ctypes.c_void_p, #returnedAlgoCount
                                                                ctypes.c_void_p] #perfResults
def cudnnFindConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc,
                                              convDesc, dxDesc,
                                              requestedAlgoCount):
    perfResultsType = cudnnConvolutionBwdDataAlgoPerf * requestedAlgoCount
    perfResults = perfResultsType()
    returnedAlgoCount = ctypes.c_int()
    status = _libcudnn.cudnnFindConvolutionBackwardDataAlgorithm(handle,
                                                                 wDesc,
                                                                 dyDesc,
                                                                 convDesc,
                                                                 dxDesc,
                                                                 ctypes.c_int(requestedAlgoCount),
                                                                 ctypes.byref(returnedAlgoCount),
                                                                 ctypes.cast(perfResults, ctypes.POINTER(cudnnConvolutionBwdDataAlgoPerf)))
    cudnnCheckStatus(status)
    return perfResults[0:returnedAlgoCount.value]

_libcudnn.cudnnGetConvolutionBackwardDataAlgorithm.restype = int
_libcudnn.cudnnGetConvolutionBackwardDataAlgorithm.argtypes = [ctypes.c_void_p,
                                                               ctypes.c_void_p,
                                                               ctypes.c_void_p,
                                                               ctypes.c_void_p,
                                                               ctypes.c_void_p,
                                                               ctypes.c_int,
                                                               ctypes.c_size_t,
                                                               ctypes.c_void_p]
def cudnnGetConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc,
                                             dxDesc, preference, memoryLimitInbytes):
    algo = ctypes.c_int()
    status = _libcudnn.cudnnGetConvolutionBackwardDataAlgorithm(handle,
                                                                wDesc,
                                                                dyDesc,
                                                                convDesc,
                                                                dxDesc,
                                                                preference,
                                                                ctypes.c_size_t(memoryLimitInbytes),
                                                                ctypes.byref(algo))
    cudnnCheckStatus(status)
    return algo

_libcudnn.cudnnGetConvolutionBackwardDataWorkspaceSize.restype = int
_libcudnn.cudnnGetConvolutionBackwardDataWorkspaceSize.argtypes = [ctypes.c_void_p,
                                                                   ctypes.c_void_p,
                                                                   ctypes.c_void_p,
                                                                   ctypes.c_void_p,
                                                                   ctypes.c_void_p,
                                                                   ctypes.c_int, #algo
                                                                   ctypes.c_void_p]
def cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc,
                                                 convDesc, dxDesc, algo):
    sizeInBytes = ctypes.c_size_t()
    status = _libcudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
                                                                    wDesc,
                                                                    dyDesc,
                                                                    convDesc,
                                                                    dxDesc,
                                                                    algo,
                                                                    ctypes.byref(sizeInBytes))
    cudnnCheckStatus(status)
    return sizeInBytes

_libcudnn.cudnnConvolutionBackwardData.restype = int
_libcudnn.cudnnConvolutionBackwardData.argtypes = [ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p,
                                                   ctypes.c_int,
                                                   ctypes.c_void_p, ctypes.c_size_t,
                                                   ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p]
def cudnnConvolutionBackwardData(handle,
                                 alpha,
                                 wDesc, w,
                                 dyDesc, dy,
                                 convDesc,
                                 algo,
                                 workspace, workSpaceSizeInBytes,
                                 beta,
                                 dxDesc, dx):
    dataType = cudnnGetTensor4dDescriptor(dyDesc)[0]
    if dataType == cudnnDataType['CUDNN_DATA_FLOAT']:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))

    status = _libcudnn.cudnnConvolutionBackwardData(handle,
                                                    alphaRef,
                                                    wDesc, w,
                                                    dyDesc, dy,
                                                    convDesc,
                                                    algo,
                                                    workspace, workSpaceSizeInBytes,
                                                    betaRef,
                                                    dxDesc, dx)
    cudnnCheckStatus(status)

class cudnnConvolutionBwdFilterAlgoPerf(ctypes.Structure):
    _fields_ = [("algo", ctypes.c_int),
                ("status", ctypes.c_int),
                ("time", ctypes.c_float),
                ("memory", ctypes.c_size_t)]

    def __str__(self):
        return '(algo=%d, status=%d, time=%f, memory=%d)' % (self.algo,
                                                             self.status,
                                                             self.time,
                                                             self.memory)
    def __repr__(self):
        return self.__str__()

_libcudnn.cudnnFindConvolutionBackwardFilterAlgorithm.restype = int
_libcudnn.cudnnFindConvolutionBackwardFilterAlgorithm.argtypes = [ctypes.c_void_p, # handle
                                                                  ctypes.c_void_p, # xDesc
                                                                  ctypes.c_void_p, # dyDesc
                                                                  ctypes.c_void_p, # convDesc
                                                                  ctypes.c_void_p, # dwDesc
                                                                  ctypes.c_int, # requestAlgoCount
                                                                  ctypes.c_void_p, #returnedAlgoCount
                                                                  ctypes.c_void_p] #perfResults
def cudnnFindConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc,
                                                convDesc, dwDesc,
                                                requestedAlgoCount):
    perfResultsType = cudnnConvolutionBwdFilterAlgoPerf * requestedAlgoCount
    perfResults = perfResultsType()
    returnedAlgoCount = ctypes.c_int()
    status = _libcudnn.cudnnFindConvolutionBackwardFilterAlgorithm(handle,
                                                                 xDesc,
                                                                 dyDesc,
                                                                 convDesc,
                                                                 dwDesc,
                                                                 ctypes.c_int(requestedAlgoCount),
                                                                 ctypes.byref(returnedAlgoCount),
                                                                 ctypes.cast(perfResults, ctypes.POINTER(cudnnConvolutionBwdFilterAlgoPerf)))
    cudnnCheckStatus(status)
    return perfResults[0:returnedAlgoCount.value]

_libcudnn.cudnnGetConvolutionBackwardFilterAlgorithm.restype = int
_libcudnn.cudnnGetConvolutionBackwardFilterAlgorithm.argtypes = [ctypes.c_void_p,
                                                                 ctypes.c_void_p,
                                                                 ctypes.c_void_p,
                                                                 ctypes.c_void_p,
                                                                 ctypes.c_void_p,
                                                                 ctypes.c_int,
                                                                 ctypes.c_size_t,
                                                                 ctypes.c_void_p]
def cudnnGetConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc,
                                               dwDesc, preference, memoryLimitInbytes):
    algo = ctypes.c_int()
    status = _libcudnn.cudnnGetConvolutionBackwardFilterAlgorithm(handle,
                                                                  xDesc,
                                                                  dyDesc,
                                                                  convDesc,
                                                                  dwDesc,
                                                                  preference,
                                                                  ctypes.c_size_t(memoryLimitInbytes),
                                                                  ctypes.byref(algo))
    cudnnCheckStatus(status)
    return algo


_libcudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize.restype = int
_libcudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize.argtypes = [ctypes.c_void_p,
                                                                     ctypes.c_void_p,
                                                                     ctypes.c_void_p,
                                                                     ctypes.c_void_p,
                                                                     ctypes.c_void_p,
                                                                     ctypes.c_int, #algo
                                                                     ctypes.c_void_p]
def cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, dyDesc,
                                                   convDesc, gradDesc, algo):
    sizeInBytes = ctypes.c_size_t()
    status = _libcudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
                                                                      xDesc,
                                                                      dyDesc,
                                                                      convDesc,
                                                                      gradDesc,
                                                                      algo,
                                                                      ctypes.byref(sizeInBytes))
    cudnnCheckStatus(status)
    return sizeInBytes

_libcudnn.cudnnConvolutionBackwardFilter.restype = int
_libcudnn.cudnnConvolutionBackwardFilter.argtypes = [ctypes.c_void_p,
                                                     ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int,
                                                     ctypes.c_void_p, ctypes.c_size_t,
                                                     ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p]
def cudnnConvolutionBackwardFilter(handle,
                                   alpha,
                                   xDesc, x,
                                   dyDesc, dy,
                                   convDesc,
                                   algo,
                                   workspace, workSpaceSizeInBytes,
                                   beta,
                                   dwDesc, dw):
    dataType = cudnnGetTensor4dDescriptor(dyDesc)[0]
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnConvolutionBackwardFilter(handle,
                                                      alphaRef,
                                                      xDesc, x,
                                                      dyDesc, dy,
                                                      convDesc,
                                                      algo,
                                                      workspace, workSpaceSizeInBytes,
                                                      betaRef,
                                                      dwDesc, dw)
    cudnnCheckStatus(status)

_libcudnn.cudnnSoftmaxForward.restype = int
_libcudnn.cudnnSoftmaxForward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p]
def cudnnSoftmaxForward(handle, algorithm, mode, alpha, srcDesc, srcData, beta, destDesc, destData):
    """"
    This routing computes the softmax function

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    algorithm : cudnnSoftmaxAlgorithm
        Enumerant to specify the softmax algorithm.
    mode : cudnnSoftmaxMode
        Enumerant to specify the softmax mode.
    alpha: float
        Scaling factor with which every element of the input tensors is multiplied.
    srcDesc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    destDesc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDesc.
    """

    dataType = cudnnGetTensor4dDescriptor(destDesc)[0]
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnSoftmaxForward(handle, algorithm, mode, alphaRef,
                                        srcDesc, srcData, betaRef,
                                        destDesc, destData)
    cudnnCheckStatus(status)

_libcudnn.cudnnSoftmaxBackward.restype = int
_libcudnn.cudnnSoftmaxBackward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p]
def cudnnSoftmaxBackward(handle, algorithm, mode, alpha, srcDesc, srcData, srcDiffDesc,
                         srcDiffData, beta, destDiffDesc, destDiffData):
    """"
    This routine computes the gradient of the softmax function.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    algorithm : cudnnSoftmaxAlgorithm
        Enumerant to specify the softmax algorithm.
    mode : cudnnSoftmaxMode
        Enumerant to specify the softmax mode.
    alpha: float
        Scaling factor with which every element of the input tensors is multiplied.
    srcDesc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc.
    srcDiffDesc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    srcDiffData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDiffData.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    destDiffDesc : cudnnTensorDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    destDiffData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDiffDesc.
    """

    dataType = cudnnGetTensor4dDescriptor(destDiffDesc)[0]
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnSoftmaxBackward(handle, algorithm, mode, alphaRef,
                                            srcDesc, srcData,
                                            srcDiffDesc, srcDiffData, betaRef,
                                            destDiffDesc, destDiffData)
    cudnnCheckStatus(status)

_libcudnn.cudnnCreatePoolingDescriptor.restype = int
_libcudnn.cudnnCreatePoolingDescriptor.argtypes = [ctypes.c_void_p]
def cudnnCreatePoolingDescriptor():
    """"
    Create pooling descriptor.

    This function creates a pooling descriptor object by allocating the memory needed to
    hold its opaque structure,

    Returns
    -------
    poolingDesc : cudnnPoolingDescriptor
        Newly allocated pooling descriptor.
    """

    poolingDesc = ctypes.c_void_p()
    status = _libcudnn.cudnnCreatePoolingDescriptor(ctypes.byref(poolingDesc))
    cudnnCheckStatus(status)

    return poolingDesc.value

_libcudnn.cudnnSetPooling2dDescriptor.restype = int
_libcudnn.cudnnSetPooling2dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int]
def cudnnSetPooling2dDescriptor(poolingDesc, mode, windowHeight, windowWidth,
                                verticalPadding, horizontalPadding, verticalStride, horizontalStride):
    """"
    Initialize a 2D pooling descriptor.

    This function initializes a previously created pooling descriptor object.

    Parameters
    ----------
    poolingDesc : cudnnPoolingDescriptor
        Handle to a previously created pooling descriptor.
    mode : cudnnPoolingMode
        Enumerant to specify the pooling mode.
    windowHeight : int
        Height of the pooling window.
    windowWidth : int
        Width of the pooling window.
    verticalPadding: int
        Size of vertical padding.
    horizontalPadding: int
        Size of horizontal padding.
    verticalStride : int
        Pooling vertical stride.
    horizontalStride : int
        Pooling horizontal stride.
    """

    status = _libcudnn.cudnnSetPooling2dDescriptor(poolingDesc, mode, windowHeight,
                                                 windowWidth, verticalPadding, horizontalPadding,
                                                 verticalStride, horizontalStride)
    cudnnCheckStatus(status)

_libcudnn.cudnnGetPooling2dDescriptor.restype = int
_libcudnn.cudnnGetPooling2dDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_void_p, ctypes.c_void_p]
def cudnnGetPooling2dDescriptor(poolingDesc):
    """"
    This function queries a previously created pooling descriptor object.

    Parameters
    ----------
    poolingDesc : cudnnPoolingDescriptor
    Handle to a previously created 2D pooling descriptor.

    Returns
    -------
    mode : cudnnPoolingMode
        Enumerant to specify the pooling mode.
    windowHeight : int
        Height of the pooling window.
    windowWidth : int
        Width of the pooling window.
    verticalPadding: int
        Size of vertical padding.
    horizontalPadding: int
        Size of horizontal padding.
    verticalStride : int
        Pooling vertical stride.
    horizontalStride : int
        Pooling horizontal stride.
    """

    mode = ctypes.c_int()
    windowHeight = ctypes.c_int()
    windowWidth = ctypes.c_int()
    verticalPadding = ctypes.c_int()
    horizontalPadding = ctypes.c_int()
    verticalStride = ctypes.c_int()
    horizontalStride = ctypes.c_int()

    status = _libcudnn.cudnnGetPooling2dDescriptor(poolingDesc, ctypes.byref(mode), ctypes.byref(windowHeight),
                                              ctypes.byref(windowWidth), ctypes.byref(verticalPadding),
                                              ctypes.byref(horizontalPadding), ctypes.byref(verticalStride),
                                              ctypes.byref(horizontalStride))
    cudnnCheckStatus(status)

    return mode.value, windowHeight.value, windowWidth.value, verticalStride.value, horizontalStride.value

_libcudnn.cudnnDestroyPoolingDescriptor.restype = int
_libcudnn.cudnnDestroyPoolingDescriptor.argtypes = [ctypes.c_void_p]
def cudnnDestroyPoolingDescriptor(poolingDesc):
    """"
    This function destroys a previously created pooling descriptor object.

    Parameters
    ----------
    poolingDesc : cudnnPoolingDescriptor
    """

    status = _libcudnn.cudnnDestroyPoolingDescriptor(poolingDesc)
    cudnnCheckStatus(status)

_libcudnn.cudnnPoolingForward.restype = int
_libcudnn.cudnnPoolingForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p]
def cudnnPoolingForward(handle, poolingDesc, alpha, srcDesc, srcData, beta, destDesc, destData):
    """"
    Perform pooling.

    This function computes pooling of input values (i.e., the maximum or average of several
    adjacent values) to produce an output with smaller height and/or width.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    poolingDesc : cudnnPoolingDescriptor
        Handle to a previously initialized pooling descriptor.
    alpha: float
        Scaling factor with which every element of the input tensor is multiplied.
    srcDesc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    destDesc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDesc.
    """

    dataType = cudnnGetTensor4dDescriptor(destDesc)[0]
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnPoolingForward(handle, poolingDesc, alphaRef,
                                        srcDesc, srcData, betaRef,
                                        destDesc, destData)
    cudnnCheckStatus(status)

_libcudnn.cudnnPoolingBackward.restype = int
_libcudnn.cudnnPoolingBackward.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p]
def cudnnPoolingBackward(handle, poolingDesc, alpha, srcDesc, srcData, srcDiffDesc,
                         srcDiffData, destDesc, destData, beta, destDiffDesc, destDiffData):
    """"
    Gradients wrt the pooling operation.

    This function computes the gradient of a pooling operation.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    poolingDesc : cudnnPoolingDescriptor
        Handle to the previously initialized pooling descriptor.
    alpha: float
        Scaling factor with which every element of the input tensors is multiplied.
    srcDesc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc.
    srcDiffDesc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    srcDiffData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDiffData.
    destDesc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDesc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    destDiffDesc : cudnnTensorDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    destDiffData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDiffDesc.
    """

    dataType = cudnnGetTensor4dDescriptor(destDesc)[0]
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnPoolingBackward(handle, poolingDesc, alphaRef,
                                            srcDesc, srcData, srcDiffDesc, srcDiffData,
                                            destDesc, destData, betaRef,
                                            destDiffDesc, destDiffData)
    cudnnCheckStatus(status)

_libcudnn.cudnnActivationForward.restype = int
_libcudnn.cudnnActivationForward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p]
def cudnnActivationForward(handle, mode, alpha, srcDesc, srcData, beta, destDesc, destData):
    """"
    Apply activation function.

    This routine applies a specified neuron activation function element-wise over each input
    value.

    In-place operation is allowed for this routine; i.e., srcData and destData pointers
    may be equal. However, this requires srcDesc and destDesc descriptors to be
    identical (particularly, the strides of the input and output must match for in-place
    operation to be allowed).

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    mode : cudnnActivationMode
        Enumerant to specify the activation mode.
    alpha: float
        Scaling factor with which every element of the input tensor is multiplied.
    srcDesc : cudnnTensor4dDescription
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation Note that if beta is zero, the output
        is not read and can contain any uninitialized data (including Nan numbers).
    destDesc : cudnnTensor4dDescription
        Handle to the previously initialized output tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDesc.
    """

    dataType = cudnnGetTensor4dDescriptor(destDesc)[0]
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnActivationForward(handle, mode, alphaRef, srcDesc, srcData,
                                              betaRef, destDesc, destData)
    cudnnCheckStatus(status)

_libcudnn.cudnnActivationBackward.restype = int
_libcudnn.cudnnActivationBackward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p]
def cudnnActivationBackward(handle, mode, alpha, srcDesc, srcData, srcDiffDesc, srcDiffData,
                            destDesc, destData, beta, destDiffDesc, destDiffData):
    """"
    Gradient of activation function.

    This routine computes the gradient of a neuron activation function.

    In-place operation is allowed for this routine; i.e., srcData and destData
    pointers may be equal and srcDiffData and destDiffData pointers may be equal.
    However, this requires the corresponding tensor descriptors to be identical
    (particularly, the strides of the input and output must match for in-place operation
    to be allowed).

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    mode : cudnnActivationMode
        Enumerant to specify the activation mode.
    alpha: float
        Scaling factor with which every element of the input tensor is multiplied.
    srcDesc : cudnnTensorDescriptor
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc.
    srcDiffDesc : cudnnTensorDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    srcDiffData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDiffData.
    destDesc : cudnnTensorDescriptor
        Handle to the previously initialized output tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDesc.
    beta: float
        Scaling factor which is applied on every element of the output tensor prior
        to adding the result of the activation gradient. Note that if beta is zero, the
        output is not read and can contain any uninitialized data (including Nan numbers).
    destDiffDesc : cudnnTensorDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    destDiffData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDiffDesc.
    """

    dataType = cudnnGetTensor4dDescriptor(destDesc)[0]
    if dataType == cudnnDataType['CUDNN_DATA_DOUBLE']:
        alphaRef = ctypes.byref(ctypes.c_double(alpha))
        betaRef = ctypes.byref(ctypes.c_double(beta))
    else:
        alphaRef = ctypes.byref(ctypes.c_float(alpha))
        betaRef = ctypes.byref(ctypes.c_float(beta))

    status = _libcudnn.cudnnActivationBackward(handle, mode, alphaRef, srcDesc, srcData,
                                               srcDiffDesc, srcDiffData,
                                               destDesc, destData, betaRef,
                                               destDiffDesc, destDiffData)
    cudnnCheckStatus(status)
