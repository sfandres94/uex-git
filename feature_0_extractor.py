#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script that extracts features from a dataset and saves them in a npy file.

Usage: feature_0_extractor.py [-h] [--dataset_split DATASET_SPLIT] [--input_size INPUT_SIZE] [--output_path OUTPUT_PATH] [--seed SEED] [--verbose]
                              input_dataset_path

Script that extracts features from a dataset and saves them in a npy file.

positional arguments:
  input_dataset_path    path to the parent directory of the dataset.

options:
  -h, --help            show this help message and exit
  --dataset_split DATASET_SPLIT, -ds DATASET_SPLIT
                        split of the dataset (default: train).
  --input_size INPUT_SIZE, -is INPUT_SIZE
                        size of the input images (default: 224).
  --output_path OUTPUT_PATH, -o OUTPUT_PATH
                        path to save the features (default: output folder in the cwd).
  --seed SEED, -s SEED  seed for the experiments (default: 42).
  --verbose, -v         provides additional details for debugging purposes.

Author: Andres J. Sanchez-Fernandez
Email: sfandres@unex.es
Date: 2024-02-20
Version: v1
"""


import argparse
import csv
import numpy as np
import os
import random
import sys
import torch
from torchinfo import summary
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        An 'argparse.Namespace' object containing the parsed arguments.
    """
    # Parser creation and description.
    parser = argparse.ArgumentParser(
        description=('Script that extracts features from a ' \
                     'dataset and saves them in a npy file.')
    )

    # Positional arguments.
    parser.add_argument('input_dataset_path', type=str,
                        help='path to the parent directory of the dataset.')

    # Options.
    parser.add_argument('--dataset_split', '-ds', type=str, default='train',
                        help='split of the dataset (default: train).')

    parser.add_argument('--input_size', '-is', type=int, default=224,
                        help='size of the input images (default: 224).')

    parser.add_argument('--output_path', '-o', type=str, default=os.path.join(os.getcwd(), 'output'),
                        help='path to save the features (default: output folder in the cwd).')

    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='seed for the experiments (default: 42).')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='provides additional details for debugging purposes.')

    return parser.parse_args(sys.argv[1:])


def set_seeds(seed: int = 42) -> None:
    """
    Set seeds for reproducibility.

    Args:
        seed (int): Seed for the random number generators (RNGs).
    """
    random.seed(seed)                                                           # Python.
    np.random.seed(seed)                                                        # RNG in other libraries.
    torch.manual_seed(seed)                                                     # PyTorch RNG.
    torch.cuda.manual_seed(seed)                                                # PyTorch on GPU.
    torch.cuda.manual_seed_all(seed)                                            # PyTorch on all GPUs.
    torch.backends.cudnn.deterministic = True                                   # Avoiding nondeterministic algorithms.
    torch.backends.cudnn.benchmark = False                                      # Causes cuDNN to deterministically select an algorithm (reduced performance).


def main(args: argparse.Namespace) -> bool:
    """
    Main function.

    Args:
        args: An 'argparse.Namespace' object containing the parsed arguments.
    
    Returns:
        A boolean indicating the success of the process.
    """
    # Convert the input dataset path to an absolute path.
    # Create the output directory if it doesn't exist.
    args.input_dataset_path = os.path.abspath(args.input_dataset_path)
    os.makedirs(args.output_path, exist_ok=True)
    dataset_name = os.path.basename(args.input_dataset_path)
    if args.verbose:
        print(f'\nDataset: {dataset_name}')

    # Convert the parsed arguments into a dict to show their values.
    if args.verbose:
        print('\nARGUMENTS')
        print('---------------------------------------------------------------')
        args_dict = vars(args)
        for arg_name in args_dict:
            arg_name_col = f'{arg_name}:'
            print(f'{arg_name_col.ljust(20)} {args_dict[arg_name]}')
        print('---------------------------------------------------------------')

    # Enable reproducibility.
    set_seeds(args.seed)

    # ImageNet normalization values.
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Define your transforms.
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        imagenet_normalize,
    ])

    # Load your dataset considering the split.
    dataset_split_path = os.path.join(args.input_dataset_path, args.dataset_split)
    if not os.path.exists(dataset_split_path):
        raise ValueError(f'The folder "{args.dataset_split}" does not exist in "{args.input_dataset_path}".')
    dataset = ImageFolder(root=dataset_split_path, transform=transform)

    # Create a DataLoader.
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if args.verbose:
        print('\nDATASET & DATALOADER')
        print('---------------------------------------------------------------')
        print(dataset)
        print(dataloader)
        print('---------------------------------------------------------------')

    # Initialize the pre-trained ResNet-50 model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = model.to(device)

    # Remove the last layer (fully connected layer) to get the features.
    model.fc = torch.nn.Identity()

    # You need to define input size to calcualte parameters.
    if args.verbose:
        print('\nMODEL')
        print('---------------------------------------------------------------')
        print(model)
        # summary(model, input_size=(batch_size, 3, args.input_size, args.input_size))
        print('---------------------------------------------------------------')

    # Extract features.
    if args.verbose:
        verbosity_step = len(dataloader) // 20
        print('\nTRAIN LOOP')
        print('---------------------------------------------------------------')
    model.eval()
    features = []
    with torch.no_grad():
        for iter, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            if args.verbose:
                if iter % verbosity_step == 0:
                    print(f'{iter}/{len(dataloader)}')
    if args.verbose:
        print('---------------------------------------------------------------')

    # Concatenate all features from all batches.
    features = np.concatenate(features, axis=0)
    if args.verbose:
        print('\nRESULTS IN .NPY FILE')
        print('---------------------------------------------------------------')
        print(features)
        print(features[0])
        print(features.shape)
        print('---------------------------------------------------------------')

    # Save features.
    output_file_path = os.path.join(args.output_path, f'{dataset_name}_{args.dataset_split}_features.npy')
    np.save(output_file_path, features)
    print(f'\n{features.shape} features saved in "{output_file_path}".\n')

    if args.verbose:
        print('SAMPLES FILE')
        print('---------------------------------------------------------------')
        samples_filename = os.path.join(args.output_path, f'{dataset_name}_{args.dataset_split}_samples_log.csv')
        with open(samples_filename, 'w', newline='') as csvfile:

            # Create a CSV writer object.
            csvwriter = csv.writer(csvfile)

            # # Write the header row.
            # csvwriter.writerow(['sample_path'])

            # Write the mapping of sample path, feature index, and cluster label for each sample.
            for sample_path, class_idx in dataset.samples:
                csvwriter.writerow([sample_path])
        print(f'Samples file saved in "{samples_filename}".')
        print('---------------------------------------------------------------\n')

    return 0


if __name__ == '__main__':
    args = get_args()                                                           # Parse and retrieve command-line arguments.
    sys.exit(main(args))                                                        # Execute the main function.
