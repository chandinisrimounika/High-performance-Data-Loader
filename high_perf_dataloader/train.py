import torch
from dataset import ImageDataset
from loader import HighPerfDataLoader

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ImageDataset("data/images")
    loader = HighPerfDataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    model = torch.nn.Conv2d(3, 8, kernel_size=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(2):
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            output = model(batch)
            loss = output.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} done")

if __name__ == "__main__":
    main()
