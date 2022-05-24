import torch
import torch.nn as nn
import torchvision.models as models

#pytorch model.save, model.load
#tensorboard; log results to tensorboard 

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """
# torchvision.models.resnet18
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 5)


#pretrained resnet 18
#redefine forward
#pytorch conv2d
#pooling in pytorch
    def forward(self, x):
        x = self.model(x)
        # x = torch.flatten(x)

        # replace the last layer
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)
        # x = self.flat(x)
        # x = F.relu(self.dense1(x)) 
        # # x = F.relu(self.dense2(x)) 
        # x = self.dense3(x) 
        # # x = self.layer1(x)
        # # x = self.ReLU(x)
        # # x = self.layer2(x)
        # # x = self.ReLU(x)
        # # x = self.layer3(x)
        return x


class Model_b(nn.Module):
    def __init__(self):
        super(Model_b, self).__init__()
        self.encoder = models.resnet18(pretrained = True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        # with torch.no_grad():
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        return self.fc(features)