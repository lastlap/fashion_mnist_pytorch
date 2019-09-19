import torch
from torch import nn
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt

PATH = Path("./model_q1.pth")

# model = nn.Sequential(nn.Linear(784,256),
#                       nn.ReLU(),
#                       nn.Linear(256,128),
#                       nn.ReLU(),
#                       nn.Linear(128,64),
#                       nn.ReLU(),
#                       nn.Linear(64,10),
#                       nn.LogSoftmax(dim=1))

model = torch.load(PATH)
model.eval()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

images, labels = next(iter(testloader))

img = images[0]
img = img.resize_(1, 784)

with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)

val=(ps==(torch.max(ps))).nonzero()[0,1]
items=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
print('Item predicted:',items[int(val)])
print(plt.imshow(img.resize_(1, 28, 28).numpy().squeeze()))
plt.show()