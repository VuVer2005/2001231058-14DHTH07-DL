import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class HockeyDataset(Dataset):
    def __init__(self, dataFolder, seqLength=15, is_train=True):
        self.dataFolder = dataFolder
        self.seqLength = seqLength
        self.classes = ['Fight', 'NonFight']
        self.spatialSequences = []
        self.labels = []

        # NÂNG CẤP: Thêm Data Augmentation cho tập Train để chống overfitting
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),          # Tăng góc xoay lên 15 độ
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Làm màu sắc biến đổi mạnh hơn
                transforms.RandomGrayscale(p=0.1),      # Thỉnh thoảng chuyển ảnh sang trắng đen
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # Làm mờ ảnh nhẹ (mô phỏng camera rung/chuyển động nhanh)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.loadSequences()

    def loadSequences(self):
        for classIdx, className in enumerate(self.classes):
            classPath = os.path.join(self.dataFolder, className)
            if not os.path.exists(classPath):
                continue

            videoFolders = [d for d in os.listdir(classPath) if os.path.isdir(os.path.join(classPath, d))]

            for videoFolder in videoFolders:
                videoFolderPath = os.path.join(classPath, videoFolder)
                allImages = sorted([img for img in os.listdir(videoFolderPath) if img.endswith('.jpg')])

                if len(allImages) >= self.seqLength:
                    selectedImages = allImages[:self.seqLength]
                    imagePaths = [os.path.join(videoFolderPath, img) for img in selectedImages]
                    self.spatialSequences.append(imagePaths)
                    self.labels.append(classIdx)

    def __len__(self):
        return len(self.spatialSequences)

    def __getitem__(self, idx):
        imagePaths = self.spatialSequences[idx]
        label = self.labels[idx]

        framesList = []
        for path in imagePaths:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            framesList.append(img)

        sequenceTensor = torch.stack(framesList)
        return sequenceTensor, label