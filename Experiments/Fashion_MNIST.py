import torch
import torchvision
import torchvision.transforms as transforms


# 加载FashionMNIST数据集
mnist_train = torchvision.datasets.FashionMNIST(root='/Datasets/FashionMNIST', train=True,
                                                download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='/Datasets/FashionMNIST', train=False,
                                               download=True, transform=transforms.ToTensor())

# 创建DataLoader实例
batch_size = 256
num_workers = 0
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers)