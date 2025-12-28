# High-performance Data Loader for Custom Dataset

## Overview

This project implements a high-performance data loading pipeline using PyTorch for image datasets.
It focuses on efficient data handling through multi-worker parallelism, asynchronous prefetching, in-memory caching, and custom image augmentations.
The project also includes benchmarking of throughput and memory usage and an example training loop.

The implementation is lightweight, cross-platform (Windows-safe).

---

## Technologies Used

* Python 3
* PyTorch
* NumPy
* OpenCV
* psutil
* Git


---

## Project Structure

The project contains:

* A custom dataset class (`ImageDataset`)
* A high-performance PyTorch DataLoader subclass
* Custom image augmentation functions
* A benchmark script to measure performance
* An example training script
* A small image dataset for demonstration

---

## Key Features

### Custom Dataset

The `ImageDataset` class loads images from disk and applies in-memory caching using an LRU strategy to avoid repeated disk reads.
Images are normalized and converted to PyTorch tensors, with augmentations applied during loading.

### High-performance DataLoader

A custom DataLoader subclass is implemented to configure performance-oriented settings such as multi-worker parallelism, asynchronous prefetching, persistent workers, and pinned memory when CUDA is available.

### Custom Augmentations

Simple augmentations such as random horizontal flip and random crop are implemented using NumPy, ensuring safe memory handling.

### Benchmarking

The benchmark script measures data loading throughput (samples per second) and memory usage using `psutil`.
Multiple runs demonstrate stable and efficient performance.

### Example Training Loop

A minimal training loop demonstrates how the custom DataLoader is used in practice with a simple convolutional model and non-blocking device transfers.

---

## How to Run

1. Install dependencies using the requirements file.
2. Place JPG or PNG images inside the `data/images` directory.
3. Run the training script to verify functionality.
4. Run the benchmark script to measure throughput and memory usage.

---

## Sample Benchmark Results

On a CPU-only Windows system, the DataLoader achieves approximately 440–485 samples per second with around 0.5–1.3 MB additional memory usage.
Minor variation between runs is expected due to system load and scheduling.

---

## Testing

Basic tests are included for validation, but running tests is optional and not required for submission.

---

## Conclusion

This project fulfills all the task requirements by implementing a PyTorch DataLoader subclass with asynchronous prefetching, multi-worker parallelism, in-memory caching, and custom augmentations.
It includes benchmark scripts and an example training loop, demonstrating a clean, efficient, and practical data loading pipeline suitable for real-world machine learning workflows.
