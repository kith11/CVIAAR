use crate::Result;
use opencv::prelude::*;
use opencv::imgproc;

pub fn preprocess_image(image: &Mat, size: (i32, i32)) -> Result<Mat> {
    let mut resized = Mat::default();
    imgproc::resize(
        image,
        &mut resized,
        opencv::core::Size::new(size.0, size.1),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    Ok(resized)
}

pub fn draw_face_info(image: &mut Mat, face: &crate::Face) -> Result<()> {
    // Draw bounding box
    let rect = opencv::core::Rect::new(
        face.bbox[0] as i32,
        face.bbox[1] as i32,
        (face.bbox[2] - face.bbox[0]) as i32,
        (face.bbox[3] - face.bbox[1]) as i32,
    );
    opencv::imgproc::rectangle(
        image,
        rect,
        opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        opencv::imgproc::LINE_8,
        0,
    )?;

    // Draw score
    let label = format!("{:.2}", face.score);
    opencv::imgproc::put_text(
        image,
        &label,
        opencv::core::Point::new(face.bbox[0] as i32, (face.bbox[1] - 10.0) as i32),
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        1,
        opencv::imgproc::LINE_AA,
        false,
    )?;

    // Draw landmarks if present
    if let Some(landmarks) = &face.landmarks {
        for p in landmarks {
            opencv::imgproc::circle(
                image,
                opencv::core::Point::new(p[0] as i32, p[1] as i32),
                2,
                opencv::core::Scalar::new(255.0, 0.0, 0.0, 0.0),
                -1,
                opencv::imgproc::LINE_AA,
                0,
            )?;
        }
    }

    Ok(())
}
