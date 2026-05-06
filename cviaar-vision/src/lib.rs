pub mod detector;
pub mod landmarker;
pub mod blink;
pub mod spoofing;
pub mod utils;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum VisionError {
    #[error("OpenCV error: {0}")]
    OpenCVError(#[from] opencv::Error),
    #[error("ORT error: {0}")]
    OrtError(#[from] ort::Error),
    #[error("Image error: {0}")]
    ImageError(#[from] image::ImageError),
    #[error("Inference error: {0}")]
    InferenceError(String),
    #[error("General error: {0}")]
    AnyhowError(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, VisionError>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Face {
    pub bbox: [f32; 4], // [x1, y1, x2, y2]
    pub score: f32,
    pub landmarks: Option<Vec<[f32; 2]>>, // 68 landmarks
    pub blink_status: Option<BlinkStatus>,
    pub liveness_score: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlinkStatus {
    pub is_blinking: bool,
    pub left_ear: f32,
    pub right_ear: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_struct() {
        let face = Face {
            bbox: [0.0, 0.0, 100.0, 100.0],
            score: 0.95,
            landmarks: None,
            blink_status: None,
            liveness_score: 1.0,
        };
        assert_eq!(face.bbox[2], 100.0);
    }

    #[test]
    fn test_blink_status_struct() {
        let blink = BlinkStatus {
            is_blinking: true,
            left_ear: 0.15,
            right_ear: 0.16,
        };
        assert!(blink.is_blinking);
    }
}
