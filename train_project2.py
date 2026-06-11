import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import HockeyDataset
from model_bilstm import CnnBiLstmModel # Import model Project 2

def trainModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Project 2 (Bi-LSTM): Đang chạy trên {device} ---")

    batchSize = 4
    learningRate = 0.0001
    numEpochs = 20
    dataFolder = "data_frames"

    print("Đang nạp dữ liệu...")
    fullDataset = HockeyDataset(dataFolder, seqLength=15, is_train=True)
    trainSize = int(0.8 * len(fullDataset))
    testSize = len(fullDataset) - trainSize
    trainDataset, testDataset = random_split(fullDataset, [trainSize, testSize])

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    model = CnnBiLstmModel(numClasses=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-3)

    print(f"Bắt đầu huấn luyện Project 2...")
    for epoch in range(numEpochs):
        model.train()
        runningLoss, correctTrains, totalTrains = 0.0, 0, 0

        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            totalTrains += labels.size(0)
            correctTrains += (predicted == labels).sum().item()

        trainAccuracy = 100 * correctTrains / totalTrains

        model.eval()
        correctTests, totalTests = 0, 0
        with torch.no_grad():
            for inputs, labels in testLoader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                totalTests += labels.size(0)
                correctTests += (predicted == labels).sum().item()

        testAccuracy = 100 * correctTests / totalTests if totalTests > 0 else 0
        print(f"Epoch [{epoch+1}/{numEpochs}] | Loss: {runningLoss/len(trainLoader):.4f} | Train Acc: {trainAccuracy:.2f}% | Test Acc: {testAccuracy:.2f}%")

    torch.save(model.state_dict(), "model_bilstm_final.pth")
    print("\n✅ Đã lưu mô hình Project 2: model_bilstm_final.pth")

if __name__ == "__main__":
    trainModel()