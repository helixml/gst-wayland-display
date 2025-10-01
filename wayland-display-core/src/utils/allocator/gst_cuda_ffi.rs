#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use gst::Buffer as GstBuffer;
use gst::ffi as gst_ffi;
use gst::glib::ffi as glib_ffi;
use gst_video::glib::translate::ToGlibPtr;
use gst_video::{VideoFormat, VideoInfoDmaDrm, VideoMeta};
use smithay::backend::allocator::Buffer;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::egl::ffi::egl::types::{EGLDisplay, EGLImageKHR, EGLint};
use std::ffi::c_void;
use std::os::fd::AsRawFd;
use std::os::raw::{c_char, c_int, c_uint};
use std::ptr;

type GstCudaContext = *mut c_void;
type GstCudaStream = *mut c_void;

#[repr(C)]
struct CUeglFrame {
    pub frame: CUeglFrameUnion,
    pub width: c_uint,
    pub height: c_uint,
    pub depth: c_uint,
    pub pitch: c_uint,
    pub plane_count: c_uint,
    pub num_channels: c_uint,
    // Followings are ENUMS
    pub frame_type: c_uint,
    pub egl_color_format: c_uint,
    pub cu_format: c_uint,
}

#[repr(C)]
union CUeglFrameUnion {
    pub p_array: [CUarray; MAX_PLANES],
    pub p_pitch: [*mut c_void; MAX_PLANES],
}

const MAX_PLANES: usize = 3;

// CUDA driver API types
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUstream = *mut c_void;
type CUdeviceptr = u64;
type CUarray = *mut c_void;
type CUgraphicsResource = *mut c_void;
type CUresult = c_uint;

// CUDA constants
const CUDA_SUCCESS: CUresult = 0;

// EGL constants
const EGL_NO_IMAGE_KHR: EGLImageKHR = std::ptr::null_mut();
const EGL_LINUX_DMA_BUF_EXT: u32 = 0x3270;
const EGL_DMA_BUF_PLANE0_FD_EXT: EGLint = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EGLint = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: EGLint = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EGLint = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EGLint = 0x3444;
const EGL_DMA_BUF_PLANE1_FD_EXT: EGLint = 0x3275;
const EGL_DMA_BUF_PLANE1_OFFSET_EXT: EGLint = 0x3276;
const EGL_DMA_BUF_PLANE1_PITCH_EXT: EGLint = 0x3277;
const EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT: EGLint = 0x3445;
const EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT: EGLint = 0x3446;
const EGL_WIDTH: EGLint = 0x3057;
const EGL_HEIGHT: EGLint = 0x3056;
const EGL_LINUX_DRM_FOURCC_EXT: EGLint = 0x3271;
const EGL_NONE: EGLint = 0x3038;

#[link(name = "cuda")]
unsafe extern "C" {
    // CUDA Driver API
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;
    fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult;
    fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;

    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn CuMemcpy2DAsync(pCopy: *const CUDA_MEMCPY2D, stream: CUstream) -> CUresult;

    // CUDA-EGL Interop
    fn cuGraphicsEGLRegisterImage(
        pCudaResource: *mut CUgraphicsResource,
        image: EGLImageKHR,
        flags: c_uint,
    ) -> CUresult;

    fn cuGraphicsUnregisterResource(resource: CUgraphicsResource) -> CUresult;

    fn cuGraphicsResourceGetMappedEglFrame(
        pEglFrame: *mut CUeglFrame,
        resource: CUgraphicsResource,
        index: c_uint,
        mipLevel: c_uint,
    ) -> CUresult;

    fn cuStreamSynchronize(stream: CUstream) -> CUresult;
}

#[link(name = "EGL")]
unsafe extern "C" {
    fn eglGetCurrentDisplay() -> EGLDisplay;
    fn eglGetProcAddress(procname: *const c_char) -> *mut c_void;
}

// EGLImage extension function pointers
type PFN_eglCreateImageKHR = unsafe extern "C" fn(
    dpy: EGLDisplay,
    ctx: *mut c_void,
    target: u32,
    buffer: *mut c_void,
    attrib_list: *const EGLint,
) -> EGLImageKHR;

type PFN_eglDestroyImageKHR = unsafe extern "C" fn(dpy: EGLDisplay, image: EGLImageKHR) -> c_int;

// CUDA memcpy2D structure
#[repr(C)]
struct CUDA_MEMCPY2D {
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

#[allow(dead_code)]
const CU_MEMORYTYPE_HOST: c_uint = 1;
#[allow(dead_code)]
const CU_MEMORYTYPE_DEVICE: c_uint = 2;
#[allow(dead_code)]
const CU_MEMORYTYPE_ARRAY: c_uint = 3;
#[allow(dead_code)]
const CU_MEMORYTYPE_UNIFIED: c_uint = 4;

// GStreamer CUDA API bindings
unsafe extern "C" {
    // gstcudaloader
    fn gst_cuda_load_library() -> glib_ffi::gboolean;

    // GstCudaContext functions
    pub fn gst_cuda_context_new(device_id: c_uint) -> *mut GstCudaContext;
    fn gst_cuda_context_get_handle(context: *mut GstCudaContext) -> CUcontext;
    fn gst_cuda_context_push(context: *mut GstCudaContext) -> glib_ffi::gboolean;
    fn gst_cuda_context_pop(pctx: *mut CUcontext) -> glib_ffi::gboolean;

    // GstCudaMemory functions
    fn gst_cuda_allocator_alloc(
        allocator: *mut gst_ffi::GstAllocator,
        context: *mut GstCudaContext,
        stream: GstCudaStream,
        info: *const gst_video::ffi::GstVideoInfo,
    ) -> *mut gst_ffi::GstMemory;

    fn gst_is_cuda_memory(mem: *mut gst_ffi::GstMemory) -> glib_ffi::gboolean;

    fn gst_cuda_memory_init_once() -> c_void;

    fn gst_cuda_stream_get_handle(stream: GstCudaStream) -> CUstream;
}

// Helper to load EGL extension functions
struct EglExtensions {
    pub create_image: PFN_eglCreateImageKHR,
    pub destroy_image: PFN_eglDestroyImageKHR,
}

impl EglExtensions {
    unsafe fn load() -> Option<Self> {
        let create_image_name = b"eglCreateImageKHR\0";
        let destroy_image_name = b"eglDestroyImageKHR\0";

        let create_image_ptr =
            unsafe { eglGetProcAddress(create_image_name.as_ptr() as *const c_char) };
        let destroy_image_ptr =
            unsafe { eglGetProcAddress(destroy_image_name.as_ptr() as *const c_char) };

        if create_image_ptr.is_null() || destroy_image_ptr.is_null() {
            return None;
        }

        Some(EglExtensions {
            create_image: unsafe { std::mem::transmute(create_image_ptr) },
            destroy_image: unsafe { std::mem::transmute(destroy_image_ptr) },
        })
    }
}

pub struct EGLImage {
    egl_extensions: EglExtensions,
    image: EGLImageKHR,
}

impl EGLImage {
    pub fn from(
        dmabuf: &Dmabuf,
        egl_display: &EGLDisplay,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Get dmabuf properties
        let width = dmabuf.width();
        let height = dmabuf.height();
        let fourcc = dmabuf.format().code as u32;

        // Get modifier if available
        let modifier: u64 = dmabuf.format().modifier.into();
        let modifier_lo = (modifier & 0xFFFFFFFF) as EGLint;
        let modifier_hi = ((modifier >> 32) & 0xFFFFFFFF) as EGLint;

        // Build EGL attribute list for DMA-BUF import
        let mut attribs = [
            EGL_WIDTH,
            width as EGLint,
            EGL_HEIGHT,
            height as EGLint,
            EGL_LINUX_DRM_FOURCC_EXT,
            fourcc as EGLint,
        ]
        .to_vec();

        let offsets = dmabuf.offsets().map(|o| o as usize).collect::<Vec<_>>();

        let strides = dmabuf.strides().map(|s| s as i32).collect::<Vec<_>>();

        for (idx, handle) in dmabuf.handles().enumerate() {
            let fd = handle.as_raw_fd();
            // Add to attribs the current plane data
            if idx == 0 {
                attribs.extend_from_slice(&[
                    EGL_DMA_BUF_PLANE0_FD_EXT,
                    fd,
                    EGL_DMA_BUF_PLANE0_OFFSET_EXT,
                    offsets[idx] as EGLint,
                    EGL_DMA_BUF_PLANE0_PITCH_EXT,
                    strides[idx],
                    EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                    modifier_lo,
                    EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                    modifier_hi,
                ]);
            } else if idx == 1 {
                attribs.extend_from_slice(&[
                    EGL_DMA_BUF_PLANE1_FD_EXT,
                    fd,
                    EGL_DMA_BUF_PLANE1_OFFSET_EXT,
                    offsets[idx] as EGLint,
                    EGL_DMA_BUF_PLANE1_PITCH_EXT,
                    strides[idx],
                    EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT,
                    modifier_lo,
                    EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT,
                    modifier_hi,
                ]);
            }
        }

        attribs.push(EGL_NONE);

        // TODO: do this once
        let egl_ext = unsafe { EglExtensions::load() }.expect("Failed to load EGL extensions");
        let egl_image = unsafe {
            (egl_ext.create_image)(
                *egl_display,
                ptr::null_mut(),
                EGL_LINUX_DMA_BUF_EXT,
                ptr::null_mut(),
                attribs.as_ptr(),
            )
        };
        if egl_image != EGL_NO_IMAGE_KHR {
            Ok(EGLImage {
                egl_extensions: egl_ext,
                image: egl_image,
            })
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to create EGLImage",
            )))
        }
    }
}

impl Drop for EGLImage {
    fn drop(&mut self) {
        unsafe {
            (self.egl_extensions.destroy_image)(eglGetCurrentDisplay(), self.image);
        }
    }
}

pub struct CUDAImage {
    cuda_graphic_resource: CUgraphicsResource,
}

impl CUDAImage {
    pub fn from(
        egl_image: &EGLImage,
        cuda_context: *mut GstCudaContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if unsafe { gst_cuda_context_push(cuda_context) } == glib_ffi::GFALSE {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to push CUDA context",
            )));
        }

        // Let's import the EGLImage into CUDA
        let mut cuda_resource: CUgraphicsResource = ptr::null_mut();
        let result = unsafe {
            cuGraphicsEGLRegisterImage(
                &mut cuda_resource,
                egl_image.image,
                0, // flags (0 = read/write)
            )
        };

        unsafe { gst_cuda_context_pop(ptr::null_mut()) };

        if result != CUDA_SUCCESS {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "Failed to register EGLImage with CUDA: {}",
                    cuda_result_to_string(result)
                ),
            )))
        } else {
            Ok(CUDAImage {
                cuda_graphic_resource: cuda_resource,
            })
        }
    }

    pub fn to_gst_buffer(
        &self,
        dma_video_info: VideoInfoDmaDrm,
        cuda_context: *mut GstCudaContext,
    ) -> Result<GstBuffer, Box<dyn std::error::Error>> {
        if unsafe { gst_cuda_context_push(cuda_context) } == glib_ffi::GFALSE {
            return Err("Failed to push CUDA context".into());
        }

        let mut egl_frame = unsafe { std::mem::zeroed() };
        let result = unsafe {
            cuGraphicsResourceGetMappedEglFrame(&mut egl_frame, self.cuda_graphic_resource, 0, 0)
        };

        unsafe { gst_cuda_context_pop(ptr::null_mut()) };

        if result != CUDA_SUCCESS {
            return Err(format!(
                "Failed to get EGL frame from CUDA resource: {}",
                cuda_result_to_string(result)
            )
            .into());
        }

        // Create a CUDA stream
        let mut video_info: gst_video::ffi::GstVideoInfo = unsafe { std::mem::zeroed() };
        unsafe { gst_video::ffi::gst_video_info_init(&mut video_info) };

        if unsafe {
            gst_video::ffi::gst_video_info_dma_drm_to_video_info(
                dma_video_info.to_glib_none().0,
                &mut video_info,
            )
        } == glib_ffi::GFALSE
        {
            return Err("Failed to convert DMA-BUF video info to GStreamer video info".into());
        }
        let stream = unsafe { std::mem::zeroed() };
        let gst_memory = unsafe {
            gst_cuda_allocator_alloc(ptr::null_mut(), cuda_context, stream, &mut video_info)
        };
        if gst_memory.is_null() {
            return Err("Failed to allocate GST CUDA memory".into());
        }
        let stream_handle = unsafe { gst_cuda_stream_get_handle(stream) };

        // Map the GStreamer memory to get destination device pointer
        let mut map_info: gst_ffi::GstMapInfo = unsafe { std::mem::zeroed() };
        let map_success =
            unsafe { gst_ffi::gst_memory_map(gst_memory, &mut map_info, gst_ffi::GST_MAP_WRITE) };

        if map_success == glib_ffi::GFALSE {
            unsafe { gst_ffi::gst_memory_unref(gst_memory) };
            return Err("Failed to map GStreamer CUDA memory".into());
        }

        let dst_device_ptr = map_info.data as CUdeviceptr;

        // Copy from EGL frame to GStreamer memory for each plane
        if unsafe { gst_cuda_context_push(cuda_context) } == glib_ffi::GFALSE {
            return Err("Failed to push CUDA context".into());
        }
        for plane in 0..egl_frame.plane_count as usize {
            let mut copy_params: CUDA_MEMCPY2D = unsafe { std::mem::zeroed() };

            // Set up source (from EGL frame)
            unsafe {
                match egl_frame.frame_type {
                    0 => {
                        // Array type
                        copy_params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                        copy_params.srcArray = egl_frame.frame.p_array[plane];
                    }
                    1 => {
                        // Pitched pointer type
                        copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                        copy_params.srcDevice = egl_frame.frame.p_pitch[plane] as CUdeviceptr;
                        copy_params.srcPitch = egl_frame.pitch as usize;
                    }
                    _ => {
                        return Err("Unsupported EGL frame type".into());
                    }
                }
            }

            copy_params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copy_params.dstDevice = dst_device_ptr + video_info.offset[plane] as u64;
            copy_params.dstPitch = video_info.stride[plane] as usize;

            // Set copy dimensions
            copy_params.WidthInBytes = video_info.stride[plane] as usize;
            copy_params.Height = match plane {
                0 => dma_video_info.height() as usize, // Y plane (or single plane)
                _ => {
                    // For YUV formats, UV planes are typically half height
                    let plane_height = match dma_video_info.format().to_string().as_str() {
                        "NV12" | "NV21" | "I420" | "YV12" => dma_video_info.height() as usize / 2,
                        _ => dma_video_info.height() as usize, // For other formats, assume same height
                    };
                    plane_height
                }
            };

            let result = unsafe { CuMemcpy2DAsync(&copy_params, stream_handle) };
            if result != CUDA_SUCCESS {
                return Err(format!(
                    "Failed to copy plane {}: {}",
                    plane,
                    cuda_result_to_string(result)
                )
                .into());
            }
        }

        let sync_result = unsafe { cuStreamSynchronize(stream_handle) };
        if sync_result != CUDA_SUCCESS {
            unsafe {
                gst_cuda_context_pop(ptr::null_mut());
                gst_ffi::gst_memory_unmap(gst_memory, &mut map_info);
                gst_ffi::gst_memory_unref(gst_memory);
            }
            return Err(format!(
                "Failed to synchronize CUDA stream: {}",
                cuda_result_to_string(sync_result)
            )
            .into());
        }

        unsafe { gst_cuda_context_pop(ptr::null_mut()) };

        // Unmap the memory
        unsafe { gst_ffi::gst_memory_unmap(gst_memory, &mut map_info) };

        // Create the buffer using GStreamer Rust bindings
        let mut buffer = gst::Buffer::new();
        {
            let buffer_ref = buffer.get_mut().unwrap();
            buffer_ref.append_memory(unsafe { gst::Memory::from_glib_full(gst_memory) });

            // Create a VideoInfo from the converted GstVideoInfo
            let video_format = match VideoFormat::from_fourcc(dma_video_info.fourcc()) {
                VideoFormat::Unknown => {
                    tracing::debug!(
                        "Failed to convert fourcc to video format: {:?}",
                        dma_video_info.fourcc()
                    );
                    VideoFormat::Bgrx // Fallback
                }
                format => format,
            };

            VideoMeta::add(
                buffer_ref,
                gst_video::VideoFrameFlags::empty(),
                video_format,
                dma_video_info.width(),
                dma_video_info.height(),
                // TODO: Add stride and offset metadata here
            )?;
        }

        Ok(buffer)
    }
}

impl Drop for CUDAImage {
    fn drop(&mut self) {
        unsafe {
            cuGraphicsUnregisterResource(self.cuda_graphic_resource);
        }
    }
}

// Error handling helper
fn cuda_result_to_string(result: CUresult) -> &'static str {
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

pub fn init_cuda() -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        static mut INITIALIZED: bool = false;
        if !INITIALIZED {
            let result = cuInit(0);
            if result == CUDA_SUCCESS {
                gst_cuda_load_library();
                gst_cuda_memory_init_once();
                INITIALIZED = true;
                Ok(())
            } else {
                Err(format!(
                    "CUDA initialization failed: {}",
                    cuda_result_to_string(result)
                )
                .into())
            }
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gst_video::glib::translate::ToGlibPtr;
    use std::ptr;

    #[test]
    fn test_cuda_device_count() {
        init_cuda().expect("Failed to initialize CUDA");

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
        init_cuda().expect("Failed to initialize CUDA");

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
        init_cuda().expect("Failed to initialize CUDA");

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
        init_cuda().expect("Failed to initialize CUDA");

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
