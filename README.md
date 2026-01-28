# Roof Detection from Satellite Images

This repository contains my solution for the dida test task.
The goal is to detect building roofs from satellite images using a neural network.

## Task
- 30 satellite images are provided
- 25 images have corresponding roof labels (binary masks)
- A neural network is trained on the labeled data
- Predictions are generated for the remaining 5 test images

## Approach
- Problem formulated as **binary image segmentation**
- Model: **U-Net**
- Loss: Binary Cross Entropy + Dice Loss
- Framework: PyTorch

## Project Structure
