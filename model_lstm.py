import torch
import torch.nn as nn

class CnnLstmModel(nn.Module):
    def __init__(self, numClasses=2, lstmHidden=128, lstmLayers=2):
        super(CnnLstmModel, self).__init__()

        # 1. Khối CNN (Giữ nguyên để đảm bảo tính công bằng khi so sánh)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Flatten() 
        )
        cnnOutputSize = 64 * 8 * 8

        # 2. Khối LSTM 1 CHIỀU
        self.lstm = nn.LSTM(
            input_size=cnnOutputSize,
            hidden_size=lstmHidden,
            num_layers=lstmLayers,
            batch_first=True
            # KHÔNG có bidirectional=True
        )

        # 3. Khối Phân loại (Kích thước đầu vào là lstmHidden)
        self.classifier = nn.Sequential(
            nn.Linear(lstmHidden, 64), 
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(64, numClasses)
        )

    def forward(self, x):
        batchSize, seqLength, c, h, w = x.size()
        x = x.view(batchSize * seqLength, c, h, w)
        cnnFeatures = self.cnn(x)
        cnnFeatures = cnnFeatures.view(batchSize, seqLength, -1)

        # Chạy qua LSTM 1 chiều
        lstmOut, _ = self.lstm(cnnFeatures)

        # Lấy đặc trưng của frame cuối cùng (đã nhìn thấy toàn bộ chuỗi)
        lastFrameOut = lstmOut[:, -1, :] 

        output = self.classifier(lastFrameOut)
        return output