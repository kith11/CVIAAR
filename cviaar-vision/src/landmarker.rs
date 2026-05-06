use crate::{Face, Result};
use ndarray::Array4;
use ort::{Session, SessionBuilder, Value};
use opencv::prelude::*;
use opencv::imgproc;

pub struct FaceLandmarker {
    session: Session,
}

impl FaceLandmarker {
    pub fn new(model_path: &str, gpu: bool) -> Result<Self> {
        let mut builder = SessionBuilder::new()?;
        if gpu {
            builder = builder.with_execution_providers([ort::ExecutionProvider::cuda()])?;
        }
        
        let session = builder.with_model_from_file(model_path)?;
        
        Ok(Self { session })
    }

    pub fn predict(&self, image: &Mat, face: &mut Face) -> Result<()> {
        // 1. Crop face region
        let x1 = face.bbox[0] as i32;
        let y1 = face.bbox[1] as i32;
        let w = (face.bbox[2] - face.bbox[0]) as i32;
        let h = (face.bbox[3] - face.bbox[1]) as i32;
        
        let mut face_roi = Mat::default();
        let roi = opencv::core::Rect::new(x1, y1, w, h);
        
        // Basic ROI bounds check
        if roi.x < 0 || roi.y < 0 || roi.x + roi.width > image.cols() || roi.y + roi.height > image.rows() {
            return Ok(()); // Or handle error
        }
        
        face_roi = Mat::roi(image, roi)?;

        // 2. Preprocessing for landmarker (e.g., 112x112 or 160x160)
        let mut resized = Mat::default();
        imgproc::resize(
            &face_roi,
            &mut resized,
            opencv::core::Size::new(112, 112),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        // Normalization
        let mut input_data = Array4::<f32>::zeros((1, 3, 112, 112));
        for y in 0..112 {
            for x in 0..112 {
                let pixel = resized.at_2d::<opencv::core::Vec3b>(y as i32, x as i32)?;
                input_data[[0, 0, y, x]] = (pixel[2] as f32 - 127.5) / 128.0;
                input_data[[0, 1, y, x]] = (pixel[1] as f32 - 127.5) / 128.0;
                input_data[[0, 2, y, x]] = (pixel[0] as f32 - 127.5) / 128.0;
            }
        }

        // 3. Inference
        let inputs = vec![Value::from_array(self.session.allocator(), &input_data)?];
        let outputs = self.session.run(inputs)?;
        
        // 4. Postprocessing: 68 points = 136 values (x, y)
        let raw_landmarks = outputs[0].try_extract::<f32>()?.view().to_owned();
        
        let mut landmarks = Vec::with_capacity(68);
        for i in 0..68 {
            let lx = raw_landmarks[[0, i * 2]] * w as f32 + x1 as f32;
            let ly = raw_landmarks[[0, i * 2 + 1]] * h as f32 + y1 as f32;
            landmarks.push([lx, ly]);
        }
        
        face.landmarks = Some(landmarks);
        
        Ok(())
    }
}
