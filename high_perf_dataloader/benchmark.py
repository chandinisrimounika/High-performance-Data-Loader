import time
import psutil
from dataset import ImageDataset
from loader import HighPerfDataLoader

def main():
    dataset = ImageDataset("data/images")
    loader = HighPerfDataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    process = psutil.Process()
    start_mem = process.memory_info().rss / 1e6
    start_time = time.time()

    num_batches = 50
    batch_size = loader.batch_size or 1

    for i, batch in enumerate(loader):
        if i == num_batches:
            break

    end_time = time.time()
    end_mem = process.memory_info().rss / 1e6

    print(f"Throughput: {(num_batches * batch_size) / (end_time - start_time):.2f} samples/sec")
    print(f"Memory used: {end_mem - start_mem:.2f} MB")

if __name__ == "__main__":
    main()
