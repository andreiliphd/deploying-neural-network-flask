from flask import Flask, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

app = Flask(__name__)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7)
        self.fc1 = nn.Linear(in_features=6144, out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@app.route('/', methods=['GET'])
def get_reply():
    model = CNN()
    pic = plt.imread('IMG_20181209_122118354.jpg')
    torch_tensor = torch.from_numpy(pic.transpose(2, 0, 1)).unsqueeze(0).float()
    response_model = model(torch_tensor)
    ali = response_model.detach().numpy().tolist()
    return jsonify(andreili=ali[0][0])


if __name__ == '__main__':
    app.run()
