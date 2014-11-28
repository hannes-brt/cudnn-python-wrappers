"""
Python interface to the NVIDIA cuDNN library
"""

import sys
import ctypes
import ctypes.util

if sys.platform in ('linux2', 'linux'):
    _libcudnn_libname_list = ['libcudnn.so', 'libcudnn.so.6.5', 'libcudnn.so.6.5.18']
elif sys.platform == 'win32':
    _libcudnn_libname_list = ['cudnn64_65.dll']
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

# Generic cuDNN error
class cudnnError(Exception):
    """cuDNN Error"""
    pass

class cudnnNotInitialized(cudnnError):
    """cuDNN library not initialized"""
    pass

class cudnnAllocFailed(cudnnError):
    """cuDNN allocation failed"""
    pass

class cudnnBadParam(cudnnError):
    """An incorrect value or parameter was passed to the function"""
    pass

class cudnnInvalidValue(cudnnError):
    """Invalid value"""
    pass

class cudnnArchMismatch(cudnnError):
    """Function requires an architectural feature absent from the device"""
    pass

class cudnnMappingError(cudnnError):
    """Access to GPU memory space failed"""
    pass

class cudnnExecutionFailed(cudnnError):
    """GPU program failed to execute"""
    pass

class cudnnInternalError(cudnnError):
    """An internal cudnn operation failed"""
    pass

class cudnnStatusNotSupported(cudnnError):
    """The functionality requested is not presently supported by cudnn"""
    pass

class cudnnStatusLicenseError(cudnnError):
    """License invalid or not found"""
    pass

cudnnExceptions = {
    1: cudnnNotInitialized,
    2: cudnnAllocFailed,
    3: cudnnBadParam,
    4: cudnnInternalError,
    5: cudnnInvalidValue,
    6: cudnnArchMismatch,
    7: cudnnMappingError,
    8: cudnnExecutionFailed,
    9: cudnnStatusNotSupported,
    10: cudnnStatusLicenseError
}

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
    'CUDNN_DATA_DOUBLE': 1  # The data is 64-bit double-precision floating point
                            # ( double ).
}

# cudnnAddMode_t is an enumerated type used by cudnnAddTensor4d() to specify how
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
    'CUDNN_CONVOLUTION': 0,
    'CUDNN_CROSS_CORRELATION': 1
}

# cudnnConvolutionPath_t is an enumerated type used by the helper routine
# cudnnGetOutputTensor4dDim() to select the results to output.
cudnnConvolutionPath = {
    'CUDNN_CONVOLUTION_FORWARD': 0, # cudnnGetOutputTensor4dDim() will return
                                    # dimensions related to the output tensor of the
                                    # forward convolution.
    'CUDNN_CONVOLUTION_WEIGHT_GRAD': 1, # cudnnGetOutputTensor4dDim() will return the
                                    # dimensions of the output filter produced while
                                    # computing the gradients, which is part of the
                                    # backward convolution.
    'CUDNN_CONVOLUTION_DATA_PATH': 2 # cudnnGetOutputTensor4dDim() will return the
                                    # dimensions of the output tensor produced while
                                    # computing the gradients, which is part of the
                                    # backward convolution.
}

# cudnnAccumulateResult_t is an enumerated type used by
# cudnnConvolutionForward() , cudnnConvolutionBackwardFilter() and
# cudnnConvolutionBackwardData() to specify whether those routines accumulate
# their results with the output tensor or simply write them to it, overwriting the previous
# value.
cudnnAccumulateResults = {
    'CUDNN_RESULT_ACCUMULATE': 0,   # The results are accumulated with (added to the
                                    # previous value of) the output tensor.
    'CUDNN_RESULT_NO_ACCUMULATE': 1 # The results overwrite the output tensor.
}

# cudnnSoftmaxAlgorithm_t is used to select an implementation of the softmax
# function used in cudnnSoftmaxForward() and cudnnSoftmaxBackward() .
cudnnSoftmaxAlgorithm = {
    'CUDNN_SOFTMAX_FAST': 0,    # This implementation applies the straightforward
                                # softmax operation.
    'CUDNN_SOFTMAX_ACCURATE': 1 # This implementation applies a scaling to the input
                                # to avoid any potential overflow.
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
    'CUDNN_POOLING_AVERAGE': 1  # The values inside the pooling window will be
                                # averaged.
}

# cudnnActivationMode_t is an enumerated type used to select the neuron activation
# function used in cudnnActivationForward() and cudnnActivationBackward() .
cudnnActivationMode = {
    'CUDNN_ACTIVATION_SIGMOID': 0,  # sigmoid function
    'CUDNN_ACTIVATION_RELU': 1,     # rectified linear function
    'CUDNN_ACTIVATION_TANH': 2      # hyperbolic tangent function
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
        try:
            raise cudnnExceptions[status]
        except KeyError:
            raise cudnnError

# Helper functions
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

_libcudnn.cudnnCreateTensor4dDescriptor.restype = int
_libcudnn.cudnnCreateTensor4dDescriptor.argtypes = [ctypes.c_void_p]
def cudnnCreateTensor4dDescriptor():
    """
    Create a Tensor4D descriptor object.

    Allocates a cudnnTensor4dDescriptor_t structure and returns a pointer to it.

    Returns
    -------
    tensor4d_descriptor : int
        Tensor4d descriptor.
    """

    tensor4d = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateTensor4dDescriptor(ctypes.byref(tensor4d))
    cudnnCheckStatus(status)
    return tensor4d.value

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
    tensorDesc : cudnnTensor4dDescriptor
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
    Initialize a Tensor4D descriptor object with strides.
    
    This function initializes a previously created Tensor4D descriptor object, similarly to
    cudnnSetTensor4dDescriptor but with the strides explicitly passed as parameters.
    This can be used to lay out the 4D tensor in any order or simply to define gaps between
    dimensions.
    
    Parameters
    ----------
    tensorDesc : cudnnTensor4dDescriptor_t
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
    Get parameters of a Tensor4D descriptor object.
    
    This function queries the parameters of the previouly initialized Tensor4D descriptor
    object.
    
    Parameters
    ----------
    tensorDesc : cudnnTensor4dDescriptor
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

_libcudnn.cudnnDestroyTensor4dDescriptor.restype = int
_libcudnn.cudnnDestroyTensor4dDescriptor.argtypes = [ctypes.c_void_p]
def cudnnDestroyTensor4dDescriptor(tensorDesc):
    """"
    Destroy a Tensor 4D descriptor.
    
    This function destroys a previously created Tensor4D descriptor object.
    
    Parameters
    ----------
    tensorDesc : cudnnTensor4dDescriptor
        Previously allocated Tensor4D descriptor object.
    """
    
    status = _libcudnn.cudnnDestroyTensor4dDescriptor(tensorDesc)
    cudnnCheckStatus(status)

_libcudnn.cudnnTransformTensor4d.restype = int
_libcudnn.cudnnTransformTensor4d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p]
def cudnnTransformTensor4d(handle, srcDesc, srcData, destDesc, destData):
    """"
    Copy data from one tensor to another.
    
    This function copies the data from one tensor to another tensor with a different
    layout. Those descriptors need to have the same dimensions but not necessarily the
    same strides. The input and output tensors must not overlap in any way (i.e., tensors
    cannot be transformed in place). This function can be used to convert a tensor with an
    unsupported format to a supported one.
    
    Parameters
    ----------
    handle : cudnnHandle
        cuDNN context.
    srcDesc : cudnnTensor4dDescriptor
        Handle to a previously initialized tensor descriptor.
    srcData : void_p
        Pointer to data of the tensor described by srcDesc descriptor.
    destDesc : cudnnTensor4dDescriptor
        Handle to a previously initialized tensor descriptor.
    destData : void_p
        Pointer to data of the tensor described by destDesc descriptor.
    """
    
    status = _libcudnn.cudnnTransformTensor4d(handle, srcDesc, srcData, destDesc, destData)
    cudnnCheckStatus(status)

_libcudnn.cudnnAddTensor4d.restype = int
_libcudnn.cudnnAddTensor4d.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p]
def cudnnAddTensor4d(handle, mode, alpha, biasDesc, biasData, srcDestDesc, srcDestData):
    """"
    Add two tensors.

    This function adds the scaled values of one tensor to another tensor. The mode parameter
    can be used to select different ways of performing the scaled addition. The amount
    of data described by the biasDesc descriptor must match exactly the amount of data
    needed to perform the addition. Therefore, the following conditions must be met:

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a cuDNN context.
    mode : cudnnAddMode
        Addition mode that describes how the addition is performed
    alpha : void_p
        Scalar factor to be applied to every data element of the bias tensor before it is added
        to the output tensor.
    biasDesc : cudnnTensor4dDescriptor
        Handle to a previoulsy initialized tensor descriptor.
    biasData : void_p
        Pointer to data of the tensor described by biasDesc.
    srcDestDesc : cudnnTensor4dDescriptor
        Handle to a previoulsy initialized tensor descriptor.
    srcDestData : void_p
        Pointer to data of the tensor described by srcDestDesc.
    """

    status = _libcudnn.cudnnAddTensor4d(handle, mode, alpha, biasDesc,
                                        biasData,
                                        srcDestDesc,
                                        srcDestData)
    cudnnCheckStatus(status)

_libcudnn.cudnnCreateFilterDescriptor.restype = int
_libcudnn.cudnnCreateFilterDescriptor.argtypes = [ctypes.c_void_p]
def cudnnCreateFilterDescriptor():
    """"
    Create a filter descriptor.
    
    This function creates a filter descriptor object by allocating the memory needed to hold
its opaque structure.
    
    Parameters
    ----------

    
    Returns
    -------
    filterDesc : cudnnFilterDescriptor
        Handle to a newly allocated filter descriptor.
    """

    filterDesc = ctypes.c_void_p()
    status = _libcudnn.cudnnCreateFilterDescriptor(ctypes.byref(filterDesc))
    cudnnCheckStatus(status)
    
    return filterDesc.value

_libcudnn.cudnnSetFilterDescriptor.restype = int
_libcudnn.cudnnSetFilterDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_int, ctypes.c_int, ctypes.c_int]
def cudnnSetFilterDescriptor(filterDesc, dataType, k, c, h, w):
    """"
    Initialize a filter descriptor.
    
    This function initializes a previously created filter descriptor object. Filters layout must
be contiguous in memory.
    
    Parameters
    ----------
    filterDesc : cudnnFilterDescriptor
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
    
    status = _libcudnn.cudnnSetFilterDescriptor(filterDesc, dataType, k, c, h, w)
    cudnnCheckStatus(status)

_libcudnn.cudnnGetFilterDescriptor.restype = int
_libcudnn.cudnnGetFilterDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cudnnGetFilterDescriptor(filterDesc):
    """"
    Get parameters of filter descriptor.
    
    This function queries the parameters of the previouly initialized filter descriptor object.
    
    Parameters
    ----------
    filterDesc : cudnnFilterDescriptor
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

    status = _libcudnn.cudnnGetFilterDescriptor(filterDesc, ctypes.byref(dataType),
                                                ctypes.byref(k), ctypes.byref(c),
                                                ctypes.byref(h), ctypes.byref(w))
    cudnnCheckStatus(status)
    
    return dataType.value, k.value, c.value, h.value, w.value

_libcudnn.cudnnDestroyFilterDescriptor.restype = int
_libcudnn.cudnnDestroyFilterDescriptor.argtypes = [ctypes.c_void_p]
def cudnnDestroyFilterDescriptor(filterDesc):
    """"
    Destroy filter descriptor.
    
    This function destroys a previously created Tensor4D descriptor object.
    
    Parameters
    ----------
    filterDesc : cudnnFilterDescriptor
    """
    
    status = _libcudnn.cudnnDestroyFilterDescriptor(filterDesc)
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

_libcudnn.cudnnSetConvolutionDescriptor.restype = int
_libcudnn.cudnnSetConvolutionDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                    ctypes.c_int]
def cudnnSetConvolutionDescriptor(convDesc, inputTensorDesc, filterDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode):
    """"
    Initialize a convolution descriptor.
    
    This function initializes a previously created convolution descriptor object, according
    to an input tensor descriptor and a filter descriptor passed as parameter. This function
    assumes that the tensor and filter descriptors corresponds to the formard convolution
    path and checks if their settings are valid. That same convolution descriptor can be
    reused in the backward path provided it corresponds to the same layer.
    
    Parameters
    ----------
    convDesc : cudnnConvolutionDescriptor
        Handle to a previously created convolution descriptor.
    inputTensorDesc : cudnnTensor4dDescriptor
        Input tensor descriptor used for that layer on the forward path.
    filterDesc : cudnnFilterDescriptor
        Filter descriptor used for that layer on the forward path.
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
    mode : int
        Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION
    """
    
    status = _libcudnn.cudnnSetConvolutionDescriptor(convDesc, inputTensorDesc, filterDesc, pad_h,
                                                     pad_w, u, v, upscalex, upscaley, mode)
    cudnnCheckStatus(status)

_libcudnn.cudnnSetConvolutionDescriptorEx.restype = int
_libcudnn.cudnnSetConvolutionDescriptorEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                      ctypes.c_int, ctypes.c_int, ctypes.c_int]
def cudnnSetConvolutionDescriptorEx(convDesc, n, c, h, w, k, r, s, pad_h, pad_w, u, v, upscalex, upscaley, mode):
    """"
    Initialize a convolution descriptor explicitely.
    
    This function initializes a previously created convolution descriptor object. It is similar
    to cudnnSetConvolutionDescriptor but every parameter of the convolution must be
    passed explicitly.
    
    Parameters
    ----------
    convDesc : cudnnConvolutionDescriptor
        Handle to a previously created convolution descriptor.
    n : int
        Number of images.
    c : int
        Number of input feature maps.
    h : int
        Height of each input feature map.
    w : int
        Width of each input feature map.
    k : int
        Number of output feature maps.
    r : int
        Height of each filter.
    s : int
        Width of each filter.
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
        Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION.
    """
    
    status = _libcudnn.cudnnSetConvolutionDescriptorEx(convDesc, n, c, h, w, k, r, s, pad_h,
                                                       pad_w, u, v, upscalex, upscaley, mode)
    cudnnCheckStatus(status)

_libcudnn.cudnnGetOutputTensor4dDim.restype = int
_libcudnn.cudnnGetOutputTensor4dDim.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cudnnGetOutputTensor4dDim(convDesc, path):
    """"
    Return the dimensions of a convolution's output.
    
    This function returns the dimensions of a convolution's output, given the convolution
descriptor and the direction of the convolution. This function can help to setup the
output tensor and allocate the proper amount of memory prior to launch the actual
convolution.
    
    Parameters
    ----------
    convDesc : cudnnConvolutionDescriptor
        Handle to a previously created convolution descriptor.
    path : cudnnConvolutionPath
        Enumerant to specify the direction of the convolution.

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

    status = _libcudnn.cudnnGetOutputTensor4dDim(convDesc, path, ctypes.byref(n),
                                                 ctypes.byref(c), ctypes.byref(h),
                                                 ctypes.byref(w))
    cudnnCheckStatus(status)
    
    return n.value, c.value, h.value, w.value

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

_libcudnn.cudnnConvolutionForward.restype = int
_libcudnn.cudnnConvolutionForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
def cudnnConvolutionForward(handle, srcDesc, srcData, filterDesc, filterData,
                            convDesc, destDesc, destData, accumulate):
    """"
    Perform forward convolution.

    This function executes convolutions or cross-correlations over src using the specified
    filters , returning results in dest.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    srcDesc : cudnnTensor4dDescriptor
        Handle to a previously initialized tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor srcDesc.
    filterDesc : cudnnFilterDescriptor
        Handle to a previously initialized filter descriptor.
    filterData : void_p
        Data pointer to GPU memory associated with the filter descriptor filterDesc.
    convDesc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    destDesc : cudnnTensor4dDescriptor
        Handle to a previously initialized tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the tensor descriptor destDesc.
    accumulate : cudnnAccumulateResult
        Enumerant that specifies whether the convolution accumulates with or
        overwrites the output tensor.
    """

    status = _libcudnn.cudnnConvolutionForward(handle, srcDesc, srcData, filterDesc, filterData,
                                               convDesc, destDesc, destData, accumulate)
    cudnnCheckStatus(status)

_libcudnn.cudnnConvolutionBackwardBias.restype = int
_libcudnn.cudnnConvolutionBackwardBias.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
def cudnnConvolutionBackwardBias(handle, srcDesc, srcData, destDesc, destData, accumulate):
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
    srcDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc .
    destDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized output tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDesc.
    accumulate : cudnnAccumulateResult
        Enumerant that specifies whether the convolution accumulates with or
        overwrites the output tensor.
    """
    
    status = _libcudnn.cudnnConvolutionBackwardBias(handle, srcDesc, srcData, destDesc,
                                                    destData, accumulate)
    cudnnCheckStatus(status)

_libcudnn.cudnnConvolutionBackwardFilter.restype = int
_libcudnn.cudnnConvolutionBackwardFilter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
def cudnnConvolutionBackwardFilter(handle, srcDesc, srcData, diffDesc, diffData,
                                   convDesc, gradDesc, gradData, accumulate):
    """"
    Compute the gradient wrt the filter coefficients.
    
    This function computes the convolution gradient with respect to the filter coefficients.
    
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    srcDesc : cudnnTensor4dDescriptor
        Handle to a previously initialized tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc .
    diffDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    diffData : void_p
        Data pointer to GPU memory associated with the input differential tensor
        descriptor diffDesc.
    convDesc : cudnnConvolutionDescriptor
        Previously initialized convolution descriptor.
    gradDesc : cudnnFilterDescriptor
        Handle to a previously initialized filter descriptor.
    gradData : void_p
        Data pointer to GPU memory associated with the filter descriptor
        gradDesc that carries the result.
    accumulate : cudnnAccumulateResult
        Enumerant that specifies whether the convolution accumulates with or
        overwrites the output tensor.
    """
    
    status = _libcudnn.cudnnConvolutionBackwardFilter(handle, srcDesc, srcData, diffDesc,
                                                      diffData, convDesc, gradDesc,
                                                      gradData, accumulate)
    cudnnCheckStatus(status)

_libcudnn.cudnnConvolutionBackwardData.restype = int
_libcudnn.cudnnConvolutionBackwardData.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
def cudnnConvolutionBackwardData(handle, filterDesc, filterData, diffDesc, diffData, convDesc,
                                 gradDesc, gradData, accumulate):
    """"
    Compute the gradients wrt the data.
    
    This function computes the convolution gradient with respect to the output tensor.
    
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    filterDesc : cudnnFilterDescriptor
        Handle to a previously initialized filter descriptor.
    filterData : void_p
        Data pointer to GPU memory associated with the filter descriptor
        filterDesc.
    diffDesc : Handle to the previously initialized input differential tensor descriptor.
    diffData,
    convDesc,
    gradDesc,
    gradData,
    accumulate
    """
    
    status = _libcudnn.cudnnConvolutionBackwardData(handle, filterDesc, filterData, diffDesc,
                                                    diffData, convDesc, gradDesc,
                                                    gradData, accumulate)
    cudnnCheckStatus(status)

_libcudnn.cudnnSoftmaxForward.restype = int
_libcudnn.cudnnSoftmaxForward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cudnnSoftmaxForward(handle, algorithm, mode, srcDesc, srcData, destDesc, destData):
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
    srcDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc .
    destDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized output tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDesc.
    """

    status = _libcudnn.cudnnSoftmaxForward(handle, algorithm, mode, srcDesc,
                                           srcData,
                                           destDesc, destData)
    cudnnCheckStatus(status)

_libcudnn.cudnnSoftmaxBackward.restype = int
_libcudnn.cudnnSoftmaxBackward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cudnnSoftmaxBackward(handle, algorithm, mode, srcDesc, srcData, srcDiffDesc,
                         srcDiffData, destDiffDesc, destDiffData):
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
    srcDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc.
    srcDiffDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    srcDiffData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDiffData.
    destDiffDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    destDiffData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDiffDesc.
    """
    
    status = _libcudnn.cudnnSoftmaxBackward(handle, algorithm, mode, srcDesc, srcData,
                                            srcDiffDesc, srcDiffData,
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

_libcudnn.cudnnSetPoolingDescriptor.restype = int
_libcudnn.cudnnSetPoolingDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int]
def cudnnSetPoolingDescriptor(poolingDesc, mode, windowHeight, windowWidth, verticalStride, horizontalStride):
    """"
    Initialize a pooling descriptor.
    
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
    verticalStride : int
        Pooling vertical stride.
    horizontalStride : int
        Pooling horizontal stride.
    """
    
    status = _libcudnn.cudnnSetPoolingDescriptor(poolingDesc, mode, windowHeight,
                                                 windowWidth, verticalStride, horizontalStride)
    cudnnCheckStatus(status)
 
_libcudnn.cudnnGetPoolingDescriptor.restype = int
_libcudnn.cudnnGetPoolingDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cudnnGetPoolingDescriptor(poolingDesc):
    """"
    This function queries a previously created pooling descriptor object.
    
    Parameters
    ----------
    poolingDesc : cudnnPoolingDescriptor
    Handle to a previously created pooling descriptor.
    
    Returns
    -------
    mode : cudnnPoolingMode
    Enumerant to specify the pooling mode.
    windowHeight : int
    Height of the pooling window.
    windowWidth : int
    Width of the pooling window.
    verticalStride : int
    Pooling vertical stride.
    horizontalStride : int
    Pooling horizontal stride.
    """
    
    mode = ctypes.c_int()
    windowHeight = ctypes.c_int()
    windowWidth = ctypes.c_int()
    verticalStride = ctypes.c_int()
    horizontalStride = ctypes.c_int()
    
    status = _libcudnn.cudnnGetPoolingDescriptor(poolingDesc, ctypes.byref(mode), ctypes.byref(windowHeight),
                                              ctypes.byref(windowWidth), ctypes.byref(verticalStride),
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
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cudnnPoolingForward(handle, poolingDesc, srcDesc, srcData, destDesc, destData):
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
    srcDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc.
    destDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized output tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDesc.
    """
    
    status = _libcudnn.cudnnPoolingForward(handle, poolingDesc, srcDesc, srcData,
                                           destDesc, destData)
    cudnnCheckStatus(status)

_libcudnn.cudnnPoolingBackward.restype = int
_libcudnn.cudnnPoolingBackward.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p]
def cudnnPoolingBackward(handle, poolingDesc, srcDesc, srcData, srcDiffDesc,
                         srcDiffData, destDesc, destData, destDiffDesc, destDiffData):
    """"
    Gradients wrt the pooling operation.
    
    This function computes the gradient of a pooling operation.
    
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    poolingDesc : cudnnPoolingDescriptor
        Handle to the previously initialized pooling descriptor.
    srcDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc.
    srcDiffDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    srcDiffData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDiffData.
    destDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized output tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDesc.
    destDiffDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    destDiffData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDiffDesc.
    """
    
    status = _libcudnn.cudnnPoolingBackward(handle, poolingDesc, srcDesc, srcData, srcDiffDesc,
                                            srcDiffData,
                                            destDesc, destData,
                                            destDiffDesc, destDiffData)
    cudnnCheckStatus(status)

_libcudnn.cudnnActivationForward.restype = int
_libcudnn.cudnnActivationForward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cudnnActivationForward(handle, mode, srcDesc, srcData, destDesc, destData):
    """"
    Apply activation function.

    This routine applies a specified neuron activation function element-wise over each input
    value.

    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    mode : cudnnActivationMode
        Enumerant to specify the activation mode.
    srcDesc : cudnnTensor4dDescription
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc.
    destDesc : cudnnTensor4dDescription
        Handle to the previously initialized output tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDesc.
    """

    status = _libcudnn.cudnnActivationForward(handle, mode, srcDesc, srcData,
                                              destDesc, destData)
    cudnnCheckStatus(status)

_libcudnn.cudnnActivationBackward.restype = int
_libcudnn.cudnnActivationBackward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def cudnnActivationBackward(handle, mode, srcDesc, srcData, srcDiffDesc, srcDiffData,
                            destDesc, destData, destDiffDesc, destDiffData):
    """"
    Gradient of activation function.
    
    This routine computes the gradient of a neuron activation function.
    
    Parameters
    ----------
    handle : cudnnHandle
        Handle to a previously created cuDNN context.
    mode : cudnnActivationMode
        Enumerant to specify the activation mode.
    srcDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized input tensor descriptor.
    srcData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDesc.
    srcDiffDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized input differential tensor descriptor.
    srcDiffData : void_p
        Data pointer to GPU memory associated with the tensor descriptor
        srcDiffData.
    destDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized output tensor descriptor.
    destData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDesc.
    destDiffDesc : cudnnTensor4dDescriptor
        Handle to the previously initialized output differential tensor descriptor.
    destDiffData : void_p
        Data pointer to GPU memory associated with the output tensor descriptor
        destDiffDesc.
    """
    
    status = _libcudnn.cudnnActivationBackward(handle, mode, srcDesc, srcData,
                                               srcDiffDesc, srcDiffData,
                                               destDesc, destData,
                                               destDiffDesc, destDiffData)
    cudnnCheckStatus(status)
