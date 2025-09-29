#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use gst::ffi as gst_ffi;
use gst::glib::ffi as glib_ffi;
use std::ffi::{c_void};
use std::os::raw::{c_char, c_int, c_uint};


// GStreamer CUDA types
#[repr(C)]
pub struct GstCudaContext {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub struct GstCudaMemory {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

#[repr(C)]
pub struct GstCudaAllocator {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

// CUDA driver API types
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUarray = *mut c_void;
pub type CUgraphicsResource = *mut c_void;
pub type CUresult = c_uint;

// EGL types
pub type EGLDisplay = *mut c_void;
pub type EGLImageKHR = *mut c_void;
pub type EGLint = i32;
pub type EGLAttrib = isize;

// CUDA constants
pub const CUDA_SUCCESS: CUresult = 0;
pub const CU_GRAPHICS_REGISTER_FLAGS_NONE: c_uint = 0x00;
pub const CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY: c_uint = 0x01;
pub const CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: c_uint = 0x02;

// EGL constants
pub const EGL_NO_IMAGE_KHR: EGLImageKHR = std::ptr::null_mut();
pub const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
pub const EGL_DMA_BUF_PLANE0_FD_EXT: EGLint = 0x3272;
pub const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EGLint = 0x3273;
pub const EGL_DMA_BUF_PLANE0_PITCH_EXT: EGLint = 0x3274;
pub const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EGLint = 0x3443;
pub const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EGLint = 0x3444;
pub const EGL_DMA_BUF_PLANE1_FD_EXT: EGLint = 0x3275;
pub const EGL_DMA_BUF_PLANE1_OFFSET_EXT: EGLint = 0x3276;
pub const EGL_DMA_BUF_PLANE1_PITCH_EXT: EGLint = 0x3277;
pub const EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT: EGLint = 0x3445;
pub const EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT: EGLint = 0x3446;
pub const EGL_WIDTH: EGLint = 0x3057;
pub const EGL_HEIGHT: EGLint = 0x3056;
pub const EGL_LINUX_DRM_FOURCC_EXT: EGLint = 0x3271;
pub const EGL_NONE: EGLint = 0x3038;

#[link(name = "cuda")]
unsafe extern "C" {
    // CUDA Driver API
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    pub fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult;
    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;

    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemcpy2D_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult;

    // CUDA-EGL Interop
    pub fn cuGraphicsEGLRegisterImage(
        pCudaResource: *mut CUgraphicsResource,
        image: EGLImageKHR,
        flags: c_uint,
    ) -> CUresult;

    pub fn cuGraphicsUnregisterResource(resource: CUgraphicsResource) -> CUresult;

    pub fn cuGraphicsResourceGetMappedEglFrame(
        pEglFrame: *mut CUeglFrame,
        resource: CUgraphicsResource,
        index: c_uint,
        mipLevel: c_uint,
    ) -> CUresult;
}

#[link(name = "EGL")]
unsafe extern "C" {
    pub fn eglGetCurrentDisplay() -> EGLDisplay;
    pub fn eglGetProcAddress(procname: *const c_char) -> *mut c_void;

    // EGLImage functions (extension, loaded via eglGetProcAddress)
    // We'll define function pointers for these
}

// EGLImage extension function pointers
pub type PFN_eglCreateImageKHR = unsafe extern "C" fn(
    dpy: EGLDisplay,
    ctx: *mut c_void,
    target: u32,
    buffer: *mut c_void,
    attrib_list: *const EGLAttrib,
) -> EGLImageKHR;

pub type PFN_eglDestroyImageKHR =
    unsafe extern "C" fn(dpy: EGLDisplay, image: EGLImageKHR) -> c_int;

// CUDA memcpy2D structure
#[repr(C)]
pub struct CUDA_MEMCPY2D {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcMemoryType: c_uint,
    pub srcHost: *const c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub srcPitch: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstMemoryType: c_uint,
    pub dstHost: *mut c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub dstPitch: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
}

// CUeglFrame structure (from CUDA EGL interop)
#[repr(C)]
pub struct CUeglFrame {
    pub frame: CUarray,
    pub width: c_uint,
    pub height: c_uint,
    pub depth: c_uint,
    pub pitch: c_uint,
    pub planeCount: c_uint,
    pub numChannels: c_uint,
    pub frameType: c_uint,
    pub eglColorFormat: c_uint,
    pub cuFormat: c_uint,
}

// GStreamer CUDA API bindings
unsafe extern "C" {
    // gstcudaloader
    pub fn gst_cuda_load_library() -> glib_ffi::gboolean;

    // GstCudaContext functions
    pub fn gst_cuda_context_new(device_id: c_uint) -> *mut GstCudaContext;
    pub fn gst_cuda_context_get_handle(context: *mut GstCudaContext) -> CUcontext;
    pub fn gst_cuda_context_push(context: *mut GstCudaContext) -> glib_ffi::gboolean;
    pub fn gst_cuda_context_pop(pctx: *mut CUcontext) -> glib_ffi::gboolean;

    // GstCudaMemory functions
    pub fn gst_cuda_allocator_alloc(
        allocator: *mut gst_ffi::GstAllocator,
        context: *mut GstCudaContext,
        stream: CUstream,
        info: *const gst_video::ffi::GstVideoInfo,
    ) -> *mut gst_ffi::GstMemory;

    pub fn gst_is_cuda_memory(mem: *mut gst_ffi::GstMemory) -> glib_ffi::gboolean;

    pub fn gst_cuda_memory_init_once() -> c_void;

    // You may need to add more based on what's actually exported
    // Check: pkg-config --cflags --libs gstreamer-cuda-1.0
}

// Helper to load EGL extension functions
pub struct EglExtensions {
    pub create_image: PFN_eglCreateImageKHR,
    pub destroy_image: PFN_eglDestroyImageKHR,
}

impl EglExtensions {
    pub unsafe fn load() -> Option<Self> {
        let create_image_name = b"eglCreateImageKHR\0";
        let destroy_image_name = b"eglDestroyImageKHR\0";

        let create_image_ptr = unsafe {eglGetProcAddress(create_image_name.as_ptr() as *const c_char)};
        let destroy_image_ptr = unsafe {eglGetProcAddress(destroy_image_name.as_ptr() as *const c_char)};

        if create_image_ptr.is_null() || destroy_image_ptr.is_null() {
            return None;
        }

        Some(EglExtensions {
            create_image: unsafe {std::mem::transmute(create_image_ptr)},
            destroy_image: unsafe {std::mem::transmute(destroy_image_ptr)},
        })
    }
}

// Error handling helper
pub fn cuda_result_to_string(result: CUresult) -> &'static str {
    match result {
        CUDA_SUCCESS => "CUDA_SUCCESS",
        1 => "CUDA_ERROR_INVALID_VALUE",
        2 => "CUDA_ERROR_OUT_OF_MEMORY",
        3 => "CUDA_ERROR_NOT_INITIALIZED",
        4 => "CUDA_ERROR_DEINITIALIZED",
        100 => "CUDA_ERROR_NO_DEVICE",
        101 => "CUDA_ERROR_INVALID_DEVICE",
        200 => "CUDA_ERROR_INVALID_IMAGE",
        201 => "CUDA_ERROR_INVALID_CONTEXT",
        _ => "CUDA_ERROR_UNKNOWN",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gst_video::glib::translate::ToGlibPtr;
    use std::ptr;

    // Helper to initialize CUDA once
    fn init_cuda() -> CUresult {
        unsafe {
            static mut INITIALIZED: bool = false;
            if !INITIALIZED {
                let result = cuInit(0);
                if result == CUDA_SUCCESS {
                    INITIALIZED = true;
                }
                result
            } else {
                CUDA_SUCCESS
            }
        }
    }

    #[test]
    fn test_cuda_init() {
        let result = init_cuda();
        assert_eq!(
            result,
            CUDA_SUCCESS,
            "CUDA initialization failed: {}",
            cuda_result_to_string(result)
        );
    }

    #[test]
    fn test_cuda_device_count() {
        init_cuda();

        unsafe {
            let mut count: c_int = 0;
            let result = cuDeviceGetCount(&mut count);
            if result == CUDA_SUCCESS {
                println!("Found {} CUDA device(s)", count);
            } else {
                println!("Skipping context test: no CUDA device available");
                return;
            }

            let mut device: CUdevice = 0;
            let result = cuDeviceGet(&mut device, 0);

            if result == CUDA_SUCCESS {
                println!("Found CUDA device: {}", device);
            } else {
                println!(
                    "No CUDA device found or error: {}",
                    cuda_result_to_string(result)
                );
            }

            // Don't fail if no device, just report
            assert!(
                result == CUDA_SUCCESS || result == 100, // 100 = NO_DEVICE
                "Unexpected error: {}",
                cuda_result_to_string(result)
            );
        }
    }

    #[test]
    fn test_cuda_context_creation() {
        init_cuda();

        unsafe {
            let mut device: CUdevice = 0;
            let result = cuDeviceGet(&mut device, 0);

            if result != CUDA_SUCCESS {
                println!("Skipping context test: no CUDA device available");
                return;
            }

            let mut ctx: CUcontext = ptr::null_mut();
            let result = cuCtxCreate_v2(&mut ctx, 0, device);

            assert_eq!(
                result,
                CUDA_SUCCESS,
                "Failed to create CUDA context: {}",
                cuda_result_to_string(result)
            );
            assert!(!ctx.is_null(), "Context should not be null");

            // Clean up
            let result = cuCtxDestroy_v2(ctx);
            assert_eq!(
                result,
                CUDA_SUCCESS,
                "Failed to destroy CUDA context: {}",
                cuda_result_to_string(result)
            );
        }
    }

    #[test]
    fn test_cuda_memory_allocation() {
        init_cuda();

        unsafe {
            let mut device: CUdevice = 0;
            if cuDeviceGet(&mut device, 0) != CUDA_SUCCESS {
                println!("Skipping memory test: no CUDA device available");
                return;
            }

            let mut ctx: CUcontext = ptr::null_mut();
            if cuCtxCreate_v2(&mut ctx, 0, device) != CUDA_SUCCESS {
                println!("Skipping memory test: failed to create context");
                return;
            }

            // Allocate 1MB
            let size = 1024 * 1024;
            let mut dptr: CUdeviceptr = 0;
            let result = cuMemAlloc_v2(&mut dptr, size);

            assert_eq!(
                result,
                CUDA_SUCCESS,
                "Failed to allocate CUDA memory: {}",
                cuda_result_to_string(result)
            );
            assert_ne!(dptr, 0, "Device pointer should not be null");

            println!(
                "Allocated {}KB at device pointer: 0x{:x}",
                size / 1024,
                dptr
            );

            // Free memory
            let result = cuMemFree_v2(dptr);
            assert_eq!(
                result,
                CUDA_SUCCESS,
                "Failed to free CUDA memory: {}",
                cuda_result_to_string(result)
            );

            // Clean up context
            cuCtxDestroy_v2(ctx);
        }
    }

    #[test]
    fn test_cuda_context_push_pop() {
        init_cuda();

        unsafe {
            let mut device: CUdevice = 0;
            if cuDeviceGet(&mut device, 0) != CUDA_SUCCESS {
                println!("Skipping context stack test: no CUDA device available");
                return;
            }

            let mut ctx: CUcontext = ptr::null_mut();
            if cuCtxCreate_v2(&mut ctx, 0, device) != CUDA_SUCCESS {
                println!("Skipping context stack test: failed to create context");
                return;
            }

            // Push context
            let result = cuCtxPushCurrent_v2(ctx);
            assert_eq!(
                result,
                CUDA_SUCCESS,
                "Failed to push context: {}",
                cuda_result_to_string(result)
            );

            // Pop context
            let mut popped_ctx: CUcontext = ptr::null_mut();
            let result = cuCtxPopCurrent_v2(&mut popped_ctx);
            assert_eq!(
                result,
                CUDA_SUCCESS,
                "Failed to pop context: {}",
                cuda_result_to_string(result)
            );
            assert_eq!(
                popped_ctx, ctx,
                "Popped context should match pushed context"
            );

            // Clean up
            cuCtxDestroy_v2(ctx);
        }
    }

    #[test]
    fn test_egl_extensions_loading() {
        unsafe {
            match EglExtensions::load() {
                Some(ext) => {
                    println!("EGL extensions loaded successfully");
                    // Verify function pointers are not null
                    assert_ne!(
                        ext.create_image as *const (),
                        ptr::null(),
                        "create_image should not be null"
                    );
                    assert_ne!(
                        ext.destroy_image as *const (),
                        ptr::null(),
                        "destroy_image should not be null"
                    );
                }
                None => {
                    println!("EGL extensions not available (may be expected)");
                }
            }
        }
    }

    #[test]
    fn test_gstreamer_cuda_context() {
        // Initialize GStreamer
        gst::init().unwrap();

        unsafe {
            if gst_cuda_load_library() == glib_ffi::GTRUE {
                println!("GStreamer CUDA library loaded successfully");
            } else {
                println!("GStreamer CUDA library not found (may not be available)");
                return;
            }
            // Try to create a GStreamer CUDA context
            let ctx = gst_cuda_context_new(0);

            if ctx.is_null() {
                println!("GStreamer CUDA context creation failed (may not be available)");
                return;
            }

            println!("GStreamer CUDA context created successfully");

            // Get the CUDA handle
            let cu_ctx = gst_cuda_context_get_handle(ctx);
            assert!(!cu_ctx.is_null(), "CUDA handle should not be null");
            println!("CUDA context handle: {:?}", cu_ctx);

            // Try to push context
            let result = gst_cuda_context_push(ctx);
            if result != 0 {
                println!("Successfully pushed CUDA context");

                // Pop it back
                let mut popped: CUcontext = ptr::null_mut();
                let pop_result = gst_cuda_context_pop(&mut popped);
                assert_ne!(pop_result, 0, "Failed to pop CUDA context");
            }
        }
    }

    #[test]
    fn test_gstreamer_cuda_memory_allocation() {
        gst::init().unwrap();

        unsafe {
            if gst_cuda_load_library() == glib_ffi::GTRUE {
                println!("GStreamer CUDA library loaded successfully");
            } else {
                println!("GStreamer CUDA library not found (may not be available)");
                return;
            }

            let ctx = gst_cuda_context_new(0);
            if ctx.is_null() {
                println!("Skipping memory allocation test: no CUDA context");
                return;
            }

            // Create a video info for a simple format
            let width = 1920;
            let height = 1080;

            let video_info =
                gst_video::VideoInfo::builder(gst_video::VideoFormat::Nv12, width, height)
                    .build()
                    .expect("Failed to build VideoInfo");


            gst_cuda_memory_init_once();
            // Try to allocate memory
            let memory = gst_cuda_allocator_alloc(
                ptr::null_mut(),
                ctx,
                ptr::null_mut(), // default stream
                video_info.to_glib_none().0,
            );

            if !memory.is_null() {
                println!("Successfully allocated CUDA memory");

                // Verify it's CUDA memory
                let is_cuda = gst_is_cuda_memory(memory);
                assert_ne!(is_cuda, 0, "Memory should be CUDA memory");

                // Clean up
                gst_ffi::gst_memory_unref(memory);
            } else {
                println!("Failed to allocate CUDA memory (may be expected)");
            }
        }
    }
}
