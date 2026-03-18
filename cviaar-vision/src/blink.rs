use crate::{Face, BlinkStatus, Result};

pub struct BlinkDetector {
    ear_threshold: f32,
    consecutive_frames: usize,
}

impl BlinkDetector {
    pub fn new(ear_threshold: f32, consecutive_frames: usize) -> Self {
        Self { ear_threshold, consecutive_frames }
    }

    pub fn calculate_ear(&self, landmarks: &[[f32; 2]]) -> (f32, f32) {
        // EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        
        let left_eye = &landmarks[36..42];
        let right_eye = &landmarks[42..48];

        let left_ear = self.eye_aspect_ratio(left_eye);
        let right_ear = self.eye_aspect_ratio(right_eye);

        (left_ear, right_ear)
    }

    fn eye_aspect_ratio(&self, eye: &[[f32; 2]]) -> f32 {
        let p1 = eye[0];
        let p2 = eye[1];
        let p3 = eye[2];
        let p4 = eye[3];
        let p5 = eye[4];
        let p6 = eye[5];

        let v1 = self.dist(p2, p6);
        let v2 = self.dist(p3, p5);
        let h1 = self.dist(p1, p4);

        (v1 + v2) / (2.0 * h1)
    }

    fn dist(&self, p1: [f32; 2], p2: [f32; 2]) -> f32 {
        ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt()
    }

    pub fn check_blink(&self, face: &mut Face) -> Result<()> {
        if let Some(landmarks) = &face.landmarks {
            let (left_ear, right_ear) = self.calculate_ear(landmarks);
            let avg_ear = (left_ear + right_ear) / 2.0;

            face.blink_status = Some(BlinkStatus {
                is_blinking: avg_ear < self.ear_threshold,
                left_ear,
                right_ear,
            });
        }
        Ok(())
    }
}
