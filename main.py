import os
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> main
import torch
import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork, Model_b
from train_functions.starting_train import starting_train


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS,
                       "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda')

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=None)
   

# Initalize dataset and model. Then train the model!
    train_dataset = StartingDataset(True)
    val_dataset = StartingDataset(False)
<<<<<<< HEAD
    samp1 = np.random.choice(len(train_dataset), size=int(0.1*len(train_dataset)), replace=False)
    samp2 = np.random.choice(len(val_dataset), size=int(0.1*len(val_dataset)), replace=False)

    trainset_1 = torch.utils.data.Subset(train_dataset, samp1)
    valset_1 = torch.utils.data.Subset(train_dataset, samp2)
    # trainset_2 = torch.utils.data.Subset(train_dataset, odds)

    # trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=4, shuffle=True, num_workers=2)
    # valloader_1 = torch.utils.data.DataLoader(valset_1, batch_size=4,
    #                                      shuffle=True, num_workers=2)
    model = StartingNetwork()
=======
    model = Model_b()
>>>>>>> main
    starting_train(
        train_dataset=trainset_1,
        val_dataset=valset_1,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )


if __name__ == "__main__":
    main()
