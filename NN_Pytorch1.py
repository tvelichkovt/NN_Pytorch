import torch
import torchvision
from torchvision import transforms, datasets

mytuple = ("hello", "world", 1) # create tuple
mylist = [3+2j, "wikipedia", "is", "cool"] # create list
mylist = mylist[:3] + ["very"] + mylist[3:] # add element to list

x = torch.Tensor([5,3])
y = torch.Tensor([2,1])

print(x*y)



train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

for data in trainset:
    print(data)
    break

X, y = data[0][0], data[1][0]

print(data[1])


import matplotlib.pyplot as plt  # pip install matplotlib

plt.imshow(data[0][0].view(28,28))
plt.show()