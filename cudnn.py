"""
Python interface to the NVIDIA cuDNN library
"""

import re
import sys
import warnings
import ctypes
import ctypes.util
import atexit
import numpy as np

from string import Template

if sys.platform == 'linux2':
    _libcudnn_libname_list = ['libcudnn.so', 'libcudnn.so.6.5', 'libcudnn.so.6.5.18']
# elif sys.platform == 'darwin':
#     _libcublas_libname_list = ['libcublas.dylib']
# elif sys.platform == 'win32':
#     _libcublas_libname_list = ['cublas64_60.dll', 'cublas32_60.dll',
#                                'cublas64_55.dll', 'cublas32_55.dll',
#                                'cublas64_50.dll', 'cublas32_50.dll']
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
if _libcudnn == None:
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
cudnnTensorFormat = {
    'CUDNN_TENSOR_NCHW': 0,
    'CUDNN_TENSOR_NHWC': 1
}

# Data type
cudnnDataType = {
    'CUDNN_DATA_FLOAT': 0,
    'CUDNN_DATA_DOUBLE': 1
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

    handle : int
        cuDNN context
    """

    handle = ctypes.c_int()
    status = _libcudnn.cudnnCreate(ctypes.byref(handle))
    cudnnCheckStatus(status)
    return handle.value

_libcudnn.cudnnDestroy.restype = int
_libcudnn.cudnnDestroy.argtypes = [ctypes.c_int]
def cudnnDestroy(handle):
    """
    Release cuDNN resources.

    Release hardware resources used by cuDNN.

    Parameters
    ----------
    handle : int
        cuDNN context.
    """

    status = _libcudnn.cudnnDestroy(ctypes.c_int(handle))
    cudnnCheckStatus(status)

_libcudnn.cudnnSetStream.restype = int
_libcudnn.cudnnSetStream.argtypes = [ctypes.c_int, ctypes.c_int]
def cudnnSetStream(handle, id):
    """
    Set current cuDNN library stream.

    Parameters
    ----------
    handle : int
        cuDNN context.
    id : int
        Stream Id.
    """

    status = _libcudnn.cudnnSetStream(handle, id)
    cudnnCheckStatus(status)

_libcudnn.cudnnGetStream.restype = int
_libcudnn.cudnnGetStream.argtypes = [ctypes.c_int, ctypes.c_void_p]
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

    id = ctypes.c_int()
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

    tensor4d = ctypes.c_int()
    status = _libcudnn.cudnnCreateTensor4dDescriptor(ctypes.byref(tensor4d))
    cudnnCheckStatus(status)
    return tensor4d.value

_libcudnn.cudnnSetTensor4dDescriptor.restype = int
_libcudnn.cudnnSetTensor4dDescriptor.argtypes = [ctypes.c_int, ctypes.c_int,
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
_libcudnn.cudnnSetTensor4dDescriptorEx.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
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
    
    status = _libcudnn.cudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride)
    cudnnCheckStatus(status)

_libcudnn.cudnnGetTensor4dDescriptor.restype = int
_libcudnn.cudnnGetTensor4dDescriptor.argtypes = 10 * [ctypes.c_int]
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

    parameters = map(ctypes.c_int, 9 * [0])

    args = [tensorDesc] + map(ctypes.byref, parameters)
    status = _libcudnn.cudnnGetTensor4dDescriptor(*args)
    cudnnCheckStatus(status)
    
    return [p.value for p in parameters]

_libcudnn.cudnnDestroyTensor4dDescriptor.restype = int
_libcudnn.cudnnDestroyTensor4dDescriptor.argtypes = [ctypes.c_int]
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
_libcudnn.cudnnTransformTensor4d.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                             ctypes.c_int, ctypes.c_void_p]
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
