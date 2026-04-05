import torch
from torchvision import datasets, transforms

class DataLoader:
    def __init__(self, data_dir, batch_size=32, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def load_data(self):
        train_dataset = datasets.ImageFolder(root=self.data_dir + '/train', transform=self.transform)
        test_dataset = datasets.ImageFolder(root=self.data_dir + '/test', transform=self.transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader
