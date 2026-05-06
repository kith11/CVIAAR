# CVIAAR-Vision Performance Benchmarks

## Overview
CVIAAR-Vision is a high-performance facial recognition and liveness detection engine written in Rust. It leverages ONNX Runtime (`ort`) and OpenCV for efficient image processing and model inference.

## Benchmarks (Target Metrics)

| Module | Target FPS (CPU) | Target FPS (GPU) | Precision (mAP) |
|---|---|---|---|
| Face Detection (UltraFace 320x240) | 50+ | 150+ | 0.90+ |
| Facial Landmarks (68-point) | 100+ | 300+ | - |
| Blink Detection (EAR) | 1000+ | - | - |
| Anti-Spoofing (Texture) | 200+ | - | - |

## System Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **CPU**: x86_64 or ARM64 (Apple Silicon supported)
- **GPU**: NVIDIA (CUDA 11.x+) or OpenCL-compatible device
- **RAM**: 4GB+ (8GB recommended for concurrent processing)

## Accuracy Metrics
- **False Acceptance Rate (FAR)**: < 0.1%
- **False Rejection Rate (FRR)**: < 1%
- **Liveness Detection**: 95%+ accuracy against printed photos and screen replays.

## Optimization Features
- **Zero-Copy Image Handling**: Minimizes memory allocation and copying between OpenCV and ONNX Runtime.
- **GPU Acceleration**: Built-in support for CUDA and OpenCL execution providers.
- **Sub-Pixel Landmark Refinement**: High-precision eye and mouth landmark localization.

## Deployment Guide
1. Ensure `onnxruntime.dll` (Windows) or `libonnxruntime.so` (Linux) is in the library path.
2. Provide the pre-trained ONNX models in the `models/` directory.
3. Use `FaceDetector` and `FaceLandmarker` for real-time processing.
