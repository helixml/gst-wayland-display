use crate::utils::allocator::gst_cuda_ffi;
use gst_video::{VideoInfo, VideoInfoDmaDrm};

#[derive(Debug, Clone)]
pub struct CUDAParams {
    pub video_info: VideoInfoDmaDrm,
    pub cuda_context: gst_cuda_ffi::CUDAContext,
}

#[derive(Debug, Clone)]
pub enum GstVideoInfo {
    RAW(VideoInfo),
    DMA(VideoInfoDmaDrm),
    CUDA(CUDAParams),
}

impl From<VideoInfo> for GstVideoInfo {
    fn from(info: VideoInfo) -> Self {
        GstVideoInfo::RAW(info)
    }
}

impl From<VideoInfoDmaDrm> for GstVideoInfo {
    fn from(info: VideoInfoDmaDrm) -> Self {
        GstVideoInfo::DMA(info)
    }
}

impl From<GstVideoInfo> for VideoInfo {
    fn from(info: GstVideoInfo) -> Self {
        match info {
            GstVideoInfo::RAW(info) => info,
            GstVideoInfo::DMA(info) => match info.to_video_info() {
                Ok(info) => info,
                Err(_) => VideoInfo::builder(info.format(), info.width(), info.height())
                    .fps(info.fps())
                    .build()
                    .expect("Failed to build VideoInfo from VideoInfoDmaDrm"),
            },
            GstVideoInfo::CUDA(params) => match params.video_info.to_video_info() {
                Ok(info) => info,
                Err(_) => VideoInfo::builder(
                    params.video_info.format(),
                    params.video_info.width(),
                    params.video_info.height(),
                )
                .fps(params.video_info.fps())
                .build()
                .expect("Failed to build VideoInfo from VideoInfoDmaDrm"),
            },
        }
    }
}
