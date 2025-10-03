use crate::cuda_call;
use ffi::{CUDA_SUCCESS, CUgraphicsResource, cuda_result_to_string};
use ffi::{GstCudaContext, PFN_eglDestroyImageKHR, eglGetProcAddress};
use gst::glib::ffi as glib_ffi;
use gst::glib::translate::ToGlibPtr;
use gst::{Buffer as GstBuffer, Element, QueryRef};
use gst_video::{VideoFormat, VideoInfoDmaDrm, VideoMeta};
use smithay::backend::allocator::Buffer;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::egl::ffi::egl::types::{EGLDisplay, EGLImageKHR, EGLint};
use std::os::fd::AsRawFd;
use std::os::raw::{c_char, c_uint};
use std::ptr;

mod ffi;

// Helper to load EGL extension functions
#[derive(Debug, Clone)]
pub struct EglExtensions {
    pub create_image: ffi::PFN_eglCreateImageKHR,
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

    pub fn new() -> Option<Self> {
        unsafe { EglExtensions::load() }
    }
}

pub struct EGLImage {
    image: EGLImageKHR,
    destroy_fn: PFN_eglDestroyImageKHR,
}

impl EGLImage {
    pub fn from(
        dmabuf: &Dmabuf,
        egl_display: &EGLDisplay,
        egl_ext: &EglExtensions,
    ) -> Result<Self, String> {
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
            ffi::EGL_WIDTH,
            width as EGLint,
            ffi::EGL_HEIGHT,
            height as EGLint,
            ffi::EGL_LINUX_DRM_FOURCC_EXT,
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
                    ffi::EGL_DMA_BUF_PLANE0_FD_EXT,
                    fd,
                    ffi::EGL_DMA_BUF_PLANE0_OFFSET_EXT,
                    offsets[idx] as EGLint,
                    ffi::EGL_DMA_BUF_PLANE0_PITCH_EXT,
                    strides[idx],
                    ffi::EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                    modifier_lo,
                    ffi::EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                    modifier_hi,
                ]);
            } else if idx == 1 {
                attribs.extend_from_slice(&[
                    ffi::EGL_DMA_BUF_PLANE1_FD_EXT,
                    fd,
                    ffi::EGL_DMA_BUF_PLANE1_OFFSET_EXT,
                    offsets[idx] as EGLint,
                    ffi::EGL_DMA_BUF_PLANE1_PITCH_EXT,
                    strides[idx],
                    ffi::EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT,
                    modifier_lo,
                    ffi::EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT,
                    modifier_hi,
                ]);
            }
        }

        attribs.push(ffi::EGL_NONE);

        let egl_image = unsafe {
            (egl_ext.create_image)(
                *egl_display,
                ptr::null_mut(),
                ffi::EGL_LINUX_DMA_BUF_EXT,
                ptr::null_mut(),
                attribs.as_ptr(),
            )
        };
        if egl_image != ffi::EGL_NO_IMAGE_KHR {
            Ok(EGLImage {
                image: egl_image,
                destroy_fn: egl_ext.destroy_image,
            })
        } else {
            Err("Failed to create EGLImage".into())
        }
    }
}

impl Drop for EGLImage {
    fn drop(&mut self) {
        unsafe {
            (self.destroy_fn)(ffi::eglGetCurrentDisplay(), self.image);
        }
    }
}

pub const CAPS_FEATURE_MEMORY_CUDA_MEMORY: &str = "memory:CUDAMemory"; // TODO: get it from FFI from gstcudamemory.h (https://github.com/GStreamer/gstreamer/blob/9d6abcc18cc9a60a212966a2daaf4a1af243f5da/subprojects/gst-plugins-bad/gst-libs/gst/cuda/gstcudamemory.h#L113-L121)

pub fn init_cuda() -> Result<(), String> {
    static mut INITIALIZED: bool = false;
    if !unsafe { INITIALIZED } {
        cuda_call!(ffi::cuInit(0))?;
        unsafe {
            ffi::gst_cuda_load_library();
            ffi::gst_cuda_memory_init_once();
            INITIALIZED = true;
        }
        Ok(())
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CUDAContext {
    ptr: *mut GstCudaContext,
}
unsafe impl Send for CUDAContext {}
unsafe impl Sync for CUDAContext {}

impl CUDAContext {
    pub fn new(device_id: c_uint) -> Result<Self, String> {
        let ptr = unsafe { ffi::gst_cuda_context_new(device_id) };
        if ptr.is_null() {
            return Err("Failed to create CUDA context".into());
        }
        Ok(CUDAContext { ptr })
    }

    pub fn as_ptr(&self) -> *mut GstCudaContext {
        self.ptr
    }
}

pub struct CUDAImage {
    cuda_graphic_resource: CUgraphicsResource,
}

impl CUDAImage {
    pub fn from(egl_image: &EGLImage, cuda_context: &CUDAContext) -> Result<Self, String> {
        let _cuda_context_guard = ffi::CudaContextGuard::new(cuda_context)?;

        // Let's import the EGLImage into CUDA
        let mut cuda_resource: CUgraphicsResource = ptr::null_mut();
        cuda_call!(ffi::cuGraphicsEGLRegisterImage(
            &mut cuda_resource,
            egl_image.image,
            0, // flags (0 = read/write)
        ))?;
        Ok(CUDAImage {
            cuda_graphic_resource: cuda_resource,
        })
    }

    pub fn to_gst_buffer(
        &self,
        dma_video_info: VideoInfoDmaDrm,
        cuda_context: &CUDAContext,
    ) -> Result<GstBuffer, Box<dyn std::error::Error>> {
        let _cuda_context_guard = ffi::CudaContextGuard::new(cuda_context)?;

        let mut egl_frame = unsafe { std::mem::zeroed() };
        cuda_call!(ffi::cuGraphicsResourceGetMappedEglFrame(
            &mut egl_frame,
            self.cuda_graphic_resource,
            0,
            0
        ))?;

        // Create Gstreamer memory
        let gst_memory =
            ffi::alloc_copy_gst_memory(egl_frame, cuda_context, dma_video_info.clone())?;

        // Create the buffer using GStreamer Rust bindings
        let mut buffer = gst::Buffer::new();
        {
            let buffer_ref = buffer.get_mut().unwrap();
            buffer_ref.append_memory(gst_memory);

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
            ffi::cuGraphicsUnregisterResource(self.cuda_graphic_resource);
        }
    }
}

pub fn gst_cuda_handle_context_query_wrapped(
    element: &Element,
    query: &mut QueryRef,
    cuda_context: &CUDAContext,
) -> bool {
    let result = unsafe {
        ffi::gst_cuda_handle_context_query(
            element.to_glib_none().0,
            query.as_mut_ptr(),
            cuda_context.ptr,
        )
    };
    result == glib_ffi::GTRUE
}
