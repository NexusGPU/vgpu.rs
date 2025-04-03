use std::os::raw::c_uint;

pub(crate) mod gpu;
pub(crate) mod mem;
pub(crate) mod nvml;

#[inline]
pub(crate) fn round_up(n: usize, base: usize) -> usize {
    if n % base != 0 {
        n + base - (n % base)
    } else {
        n
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(crate) struct CufuncSt {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(crate) struct CustreamSt {
    _unused: [u8; 0],
}

pub(crate) type CUstream = *mut CustreamSt;
pub(crate) type CUfunction = *mut CufuncSt;
pub(crate) type CUresult = c_uint;

pub(crate) type CUdeviceptrV2 = std::ffi::c_ulonglong;
pub(crate) type CUdeviceptr = CUdeviceptrV2;

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct CUdeviceptrV1(pub c_uint);

#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) struct CudaArray3dDescriptorSt {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
    pub format: CuarrayFormatEnum,
    pub num_channels: u64,
    pub flags: u64,
}

pub(crate) type CudaArray3dDescriptorV2 = CudaArray3dDescriptorSt;
pub(crate) type CudaArray3dDescriptor = CudaArray3dDescriptorV2;

#[allow(dead_code)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) enum CuarrayFormatEnum {
    CuAdFormatUnsignedInt8 = 1,
    CuAdFormatUnsignedInt16 = 2,
    CuAdFormatUnsignedInt32 = 3,
    CuAdFormatSignedInt8 = 8,
    CuAdFormatSignedInt16 = 9,
    CuAdFormatSignedInt32 = 10,
    CuAdFormatHalf = 16,
    CuAdFormatFloat = 32,
    CuAdFormatNv12 = 176,
    CuAdFormatUnormInt8x1 = 192,
    CuAdFormatUnormInt8x2 = 193,
    CuAdFormatUnormInt8x4 = 194,
    CuAdFormatUnormInt16x1 = 195,
    CuAdFormatUnormInt16x2 = 196,
    CuAdFormatUnormInt16x4 = 197,
    CuAdFormatSnormInt8x1 = 198,
    CuAdFormatSnormInt8x2 = 199,
    CuAdFormatSnormInt8x4 = 200,
    CuAdFormatSnormInt16x1 = 201,
    CuAdFormatSnormInt16x2 = 202,
    CuAdFormatSnormInt16x4 = 203,
    CuAdFormatBc1Unorm = 145,
    CuAdFormatBc1UnormSrgb = 146,
    CuAdFormatBc2Unorm = 147,
    CuAdFormatBc2UnormSrgb = 148,
    CuAdFormatBc3Unorm = 149,
    CuAdFormatBc3UnormSrgb = 150,
    CuAdFormatBc4Unorm = 151,
    CuAdFormatBc4Snorm = 152,
    CuAdFormatBc5Unorm = 153,
    CuAdFormatBc5Snorm = 154,
    CuAdFormatBc6hUf16 = 155,
    CuAdFormatBc6hSf16 = 156,
    CuAdFormatBc7Unorm = 157,
    CuAdFormatBc7UnormSrgb = 158,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(crate) struct CumipmappedArraySt {
    _unused: [u8; 0],
}
pub(crate) type CUmipmappedArray = *mut CumipmappedArraySt;

pub(crate) type CUdevice = std::ffi::c_int;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(crate) struct CuarraySt {
    _unused: [u8; 0],
}
pub(crate) type CUarray = *mut CuarraySt;

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct CudaArrayDescriptorSt {
    pub width: u64,
    pub height: u64,
    pub format: CuarrayFormatEnum,
    pub num_channels: u64,
}
pub(crate) type CudaArrayDescriptor = CudaArrayDescriptorSt;

// Return codes for NVML functions
pub(crate) type NvmlReturnT = c_uint;
pub(crate) const NVML_SUCCESS: NvmlReturnT = 0;
pub(crate) const NVML_ERROR_UNKNOWN: NvmlReturnT = 999;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(crate) struct NvmlDeviceSt {
    _unused: [u8; 0],
}
pub(crate) type NvmlDeviceT = *mut NvmlDeviceSt;
