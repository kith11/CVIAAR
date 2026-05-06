use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cviaar_vision::detector::FaceDetector;
use opencv::imgcodecs;

fn criterion_benchmark(c: &mut Criterion) {
    // Note: This benchmark requires the UltraFace ONNX model to be present at the specified path.
    // Replace with a valid path for testing.
    let model_path = "models/ultraface.onnx";
    
    // Load a sample image
    let image = imgcodecs::imread("data/faces/25/1.jpg", imgcodecs::IMREAD_COLOR).unwrap();

    // Check if the model exists before benchmarking
    if std::path::Path::new(model_path).exists() {
        let detector = FaceDetector::new(model_path, false).unwrap();
        
        c.bench_function("face_detection", |b| {
            b.iter(|| {
                detector.detect(black_box(&image)).unwrap();
            })
        });
    } else {
        println!("Warning: Model file not found, skipping benchmark.");
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
