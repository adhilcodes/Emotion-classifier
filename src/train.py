import torch
from torch.utils.data import DataLoader
from model import EmotionCNN
from data_loader import EmotionData
from sklearn.metrics import accuracy_score
import pandas as pd

def training_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
       
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_outputs, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_outputs.append(outputs)
                val_labels.append(labels)

        val_outputs = torch.cat(val_outputs)
        val_labels = torch.cat(val_labels)
        _, preds = torch.max(val_outputs, 1)
        accuracy = accuracy_score(val_labels.cpu().numpy(), preds.cpu().numpy())

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {accuracy:.4f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), './models/best_model.pth')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 60
    learning_rate = 0.001
	
    train_df = pd.read_csv("./data/train.csv", usecols=['image', 'label'])
    val_df = pd.read_csv("./data/val.csv", usecols=['image', 'label'])
    
    train_dataset = EmotionData(train_df)
    val_dataset = EmotionData(val_df)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = EmotionCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_loop(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=epochs)

