import torch
import torch.nn as nn

class CnnBiLstmModel(nn.Module):
    def __init__(self, numClasses=2, lstmHidden=128, lstmLayers=2):
        super(CnnBiLstmModel, self).__init__()

        # 1. Khối CNN (Giống hệt Project 1 để so sánh công bằng)
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

        # 2. Khối BI-LSTM 2 CHIỀU
        self.lstm = nn.LSTM(
            input_size=cnnOutputSize,
            hidden_size=lstmHidden,
            num_layers=lstmLayers,
            batch_first=True,
            bidirectional=True  # BẬT 2 CHIỀU
        )

        # 3. Khối Phân loại (Kích thước đầu vào là lstmHidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(lstmHidden * 2, 64), 
            nn.ReLU(),
            nn.Dropout(0.5),  # Tăng Dropout lên 0.4 để chống overfitting mạnh hơn cho Bi-LSTM
            nn.Linear(64, numClasses)
        )

    def forward(self, x):
        batchSize, seqLength, c, h, w = x.size()
        x = x.view(batchSize * seqLength, c, h, w)
        cnnFeatures = self.cnn(x)
        cnnFeatures = cnnFeatures.view(batchSize, seqLength, -1)

        # Chạy qua Bi-LSTM, lấy thêm h_n (hidden state)
        lstmOut, (h_n, c_n) = self.lstm(cnnFeatures)

        # --- LOGIC CHUẨN CHO BI-LSTM ---
        # h_n có shape: (num_layers * 2, batch, hidden)
        # Lấy 2 layer cuối cùng (Index -2 là Forward, Index -1 là Backward của layer cuối)
        last_layer_hn = h_n[-2:, :, :] 
        
        # Đổi chiều và duỗi thẳng thành (batch, hidden * 2)
        lastFrameOut = last_layer_hn.transpose(0, 1).contiguous().view(batchSize, -1)

        output = self.classifier(lastFrameOut)
        return output