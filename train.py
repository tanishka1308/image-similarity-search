import torch
from torch.utils.data import DataLoader
from src.model import ImageEmbeddingNet
from src.dataset import RobustSyntheticTripletDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    dataset = RobustSyntheticTripletDataset(image_dir="dataset\images")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    model = ImageEmbeddingNet(embedding_dim=256)
    model.to(device)

    criterion = torch.nn.TripletMarginLoss(margin=0.2, p=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            ea = model(anchor)
            ep = model(positive)
            en = model(negative)

            loss = criterion(ea, ep, en)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {running_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "model_weights.pth")
    print("Model weights saved to model_weights.pth")

if __name__ == "__main__":
    main()