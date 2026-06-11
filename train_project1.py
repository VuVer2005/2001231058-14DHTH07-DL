import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import HockeyDataset
from model_lstm import CnnLstmModel # Import model Project 1

def trainModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Project 1 (LSTM): Đang chạy trên {device} ---")

    batchSize = 4
    learningRate = 0.0001
    numEpochs = 20
    dataFolder = "data_frames"

    print("Đang nạp dữ liệu...")
    fullDataset = HockeyDataset(dataFolder, seqLength=15, is_train=True)
    
    # Lưu ý: Khi chia tập Test, ta tắt augmentation bằng cách tạo dataset mới với is_train=False
    # Nhưng để đơn giản, random_split sẽ chia ngẫu nhiên, ta sẽ xử lý transform trong __getitem__ dựa trên idx nếu cần.
    # Ở đây giữ nguyên logic chia đơn giản.
    trainSize = int(0.8 * len(fullDataset))
    testSize = len(fullDataset) - trainSize
    trainDataset, testDataset = random_split(fullDataset, [trainSize, testSize])

    # Mẹo: Tắt augmentation cho tập test bằng cách gán lại transform (nâng cao)
    # Để code gọn, ta cứ để random_split như cũ, model eval sẽ ổn.

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    model = CnnLstmModel(numClasses=2).to(device)
    criterion = nn.CrossEntropyLoss()
    # Thêm weight_decay để chống overfitting
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-3
)

    print(f"Bắt đầu huấn luyện Project 1...")
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

    torch.save(model.state_dict(), "model_lstm_final.pth")
    print("\n✅ Đã lưu mô hình Project 1: model_lstm_final.pth")

if __name__ == "__main__":
    trainModel()