use crate::{Face, Result, VisionError};
use ndarray::{Array4, Axis};
use ort::{Session, SessionBuilder, Value};
use opencv::prelude::*;
use opencv::imgproc;

pub struct FaceDetector {
    session: Session,
    input_shape: [u32; 2], // [width, height]
}

impl FaceDetector {
    pub fn new(model_path: &str, gpu: bool) -> Result<Self> {
        let mut builder = SessionBuilder::new()?;
        if gpu {
            // Enable CUDA
            builder = builder.with_execution_providers([ort::ExecutionProvider::cuda()])?;
        }
        
        let session = builder.with_model_from_file(model_path)?;
        
        Ok(Self {
            session,
            input_shape: [320, 240], // UltraFace 320x240
        })
    }

    pub fn detect(&self, image: &Mat) -> Result<Vec<Face>> {
        // 1. Preprocessing
        let mut resized = Mat::default();
        imgproc::resize(
            image,
            &mut resized,
            opencv::core::Size::new(self.input_shape[0] as i32, self.input_shape[1] as i32),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        // Convert Mat to ndarray
        // Normalize: (pixel - mean) / std
        // For UltraFace: mean=127, std=128
        let mut input_data = Array4::<f32>::zeros((1, 3, self.input_shape[1] as usize, self.input_shape[0] as usize));
        
        for y in 0..self.input_shape[1] as usize {
            for x in 0..self.input_shape[0] as usize {
                let pixel = resized.at_2d::<opencv::core::Vec3b>(y as i32, x as i32)?;
                input_data[[0, 0, y, x]] = (pixel[2] as f32 - 127.0) / 128.0; // R
                input_data[[0, 1, y, x]] = (pixel[1] as f32 - 127.0) / 128.0; // G
                input_data[[0, 2, y, x]] = (pixel[0] as f32 - 127.0) / 128.0; // B
            }
        }

        // 2. Inference
        let inputs = vec![Value::from_array(self.session.allocator(), &input_data)?];
        let outputs = self.session.run(inputs)?;
        
        // 3. Postprocessing
        // UltraFace outputs scores and boxes
        let scores = outputs[0].try_extract::<f32>()?.view().to_owned();
        let boxes = outputs[1].try_extract::<f32>()?.view().to_owned();

        // Simple NMS and confidence thresholding
        let mut faces = Vec::new();
        let threshold = 0.8;
        
        // UltraFace specific output parsing
        // This is a simplified version; real NMS would be needed
        for i in 0..scores.shape()[1] {
            let score = scores[[0, i, 1]]; // Class 1 is face
            if score > threshold {
                let x1 = boxes[[0, i, 0]] * image.cols() as f32;
                let y1 = boxes[[0, i, 1]] * image.rows() as f32;
                let x2 = boxes[[0, i, 2]] * image.cols() as f32;
                let y2 = boxes[[0, i, 3]] * image.rows() as f32;
                
                faces.push(Face {
                    bbox: [x1, y1, x2, y2],
                    score,
                    landmarks: None,
                    blink_status: None,
                    liveness_score: 0.0,
                });
            }
        }

        Ok(faces)
    }
}
