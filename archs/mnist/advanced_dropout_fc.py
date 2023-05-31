import torch
import torch.nn as nn
import torch.nn.functional as F
class advanced_drop_fc(nn.Module):

    def __init__(self, num_classes=10):
        super(advanced_drop_fc, self).__init__()
        self.fc1 = nn.Linear(28*28, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, num_classes)

        # self.classifier = nn.Sequential(
        #     nn.Linear(28*28, 300),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(300, 100),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(100, num_classes),
        # )

    def forward(self, x):
        x = torch.flatten(x, 1)
        # x = self.classifier(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
