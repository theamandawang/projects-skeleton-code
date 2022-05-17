import torch
import torchvision
import numpy
import random
import constants
from PIL import Image
import pandas
#from resizeimage import resizeimage
import matplotlib.pyplot as plt
# from matplotlib import image
# from matplotlib import pyplot
class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """
    ###create this function

    def __init__(self, isTrain, datapath = constants.PATH_TO_DATA):
        self.datapath = datapath
        if(isTrain):
            self.csv_data = pandas.read_csv(self.datapath + 'train.csv').head(19257).to_numpy()
        else:
            self.csv_data = pandas.read_csv(self.datapath + 'train.csv').tail(2140).to_numpy()
    def __getitem__(self, index):
        ## do loading here
        image_name, label = self.csv_data[index]
        funcs = [torchvision.transforms.CenterCrop(100), torchvision.transforms.RandomRotation(50, expand=True)]
        i = random.randint(0, 2)
        # transform image
        # save image
        with Image.open(constants.PATH_TO_DATA+'/train_images/'+image_name) as inputs:
            if(self.isTrain):
                if(i < 2):
                    inputs = funcs[i](inputs)
                else:
                    pass
            inputs = torchvision.transforms.functional.resize(inputs, (224, 224))
            inputs = torchvision.transforms.ToTensor()(inputs)
            return inputs, label

    def countTypes(self):
        df = pandas.DataFrame(data=self.csv_data, columns=['id', 'label'])
        df['label']=df['label'].astype(int)
        df.hist(column=['label'])
        plt.show()
        return 

    def __len__(self):
        return len(self.csv_data)
