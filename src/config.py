import torch

# Pytorch install: https://pytorch.org/get-started/locally/

# untagged raw images, and destination for modified images
FILE_IN_PATH = 'F:/Python Projects/PyTorch-Valve-v1/model_pictures/originals_untagged/'
FILE_OUT_PATH = 'F:/Python Projects/PyTorch-Valve-v1/model_pictures/originals_modified/'

# input and output for inference
VALIDATE_SOURCE = '../model_test/source'
VALIDATE_PREDICTIONS = '../model_test/predictions'

# paths for model training
TRAIN_DIR = '../model_pictures/train'            # training images and XML files directory
MODEL_LOCATION = '../model_outputs/model100.pth'
VALID_DIR = '../model_pictures/test'              # validation images and XML files directory

# computational setup
COMPUTATION_DEVICE = 'cuda'
DEVICE = torch.device(COMPUTATION_DEVICE)
BATCH_SIZE = 9      # increase / decrease according to GPU memory - 9 works for GeForce 1070, 8GB RAM
RESIZE_TO = 640     # resize the image for training and transforms - TODO: how big can we go?  is bigger better?
NUM_EPOCHS = 1000    # number of epochs to train for
# TODO: how do epochs relate to training quality (output graphs) - other parameters?
NUM_CLASSES = 2

LEARNING_RATE_START = 0.0002     # default 0.001
LEARNING_RATE_DECAY = 0.98      # starting point for exponential decay of learning rate
MOMENTUM_START = 0.90           # default 0.9
WEIGHT_DECAY = 0.0005           # default 0.0005

# classes: 0 index is reserved for background
NUM_CLASSES = 2
CLASSES = ['background', 'valve']


# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = '../model_outputs/'
SAVE_PLOTS_EPOCH = 250  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 100   # save model after these many epochs
