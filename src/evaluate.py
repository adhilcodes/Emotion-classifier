import torch
from torch.utils.data import DataLoader
from model import EmotionCNN
from data_loader import EmotionData
from sklearn.metrics import accuracy_score

def evaluate_model(model, test_loader):
    model.eval()
    test_outputs = []
    test_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_outputs.append(outputs.cpu())
            test_labels.append(labels.cpu())

    test_outputs = torch.cat(test_outputs)
    test_labels = torch.cat(test_labels)
    _, preds = torch.max(test_outputs, 1)
    accuracy = accuracy_score(test_labels.cpu().numpy(), preds.cpu().numpy())
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load('./models/best_model.pth'))

    test_dataset = EmotionData('./data/test.csv')
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    evaluate_model(model, test_dataloader)

