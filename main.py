import os
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
    model = Model_b()
    starting_train(
        train_dataset=trainset_1,
        val_dataset=valset_1,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )


if __name__ == "__main__":
    main()
