use crate::{Face, Result};
use opencv::prelude::*;
use opencv::imgproc;

pub struct AntiSpoofing;

impl AntiSpoofing {
    pub fn new() -> Self {
        Self
    }

    pub fn check_liveness(&self, image: &Mat, face: &mut Face) -> Result<()> {
        // Simple texture analysis for spoofing detection
        // Crop the face region
        let x1 = face.bbox[0] as i32;
        let y1 = face.bbox[1] as i32;
        let w = (face.bbox[2] - face.bbox[0]) as i32;
        let h = (face.bbox[3] - face.bbox[1]) as i32;
        
        let roi = opencv::core::Rect::new(x1, y1, w, h);
        if roi.x < 0 || roi.y < 0 || roi.x + roi.width > image.cols() || roi.y + roi.height > image.rows() {
            return Ok(());
        }
        
        let mut face_roi = Mat::roi(image, roi)?;
        let mut gray = Mat::default();
        imgproc::cvt_color(&face_roi, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Simple Laplacian variance as a proxy for sharpness (texture detail)
        let mut laplacian = Mat::default();
        imgproc::laplacian(&gray, &mut laplacian, opencv::core::CV_64F, 1, 1.0, 0.0, opencv::core::BORDER_DEFAULT)?;
        
        let mean = opencv::core::mean(&laplacian, &Mat::default())?;
        let mut stddev = Mat::default();
        opencv::core::mean_std_dev(&laplacian, &mut mean.to_owned(), &mut stddev, &Mat::default())?;
        
        let variance = stddev.at_2d::<f64>(0, 0)? * stddev.at_2d::<f64>(0, 0)?;
        
        // Normalize liveness score: low variance often indicates a printed photo or screen
        let liveness_score = if variance > 100.0 {
            1.0
        } else {
            variance / 100.0
        };

        face.liveness_score = liveness_score as f32;

        Ok(())
    }
}
