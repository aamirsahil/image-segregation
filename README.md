# Image Segregation and Ranking Model

This repository contains an image segregation model that takes in a list of images and ranks them based on provided conditions. The model processes images from the `input` folder and outputs the ranked images into the `outputs` folder, with each image placed in a subfolder corresponding to its rank (1-5).

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Input and Output Structure](#input-and-output-structure)
- [Custom Conditions](#custom-conditions)

## Overview

The model is designed to segregate and rank images based on user-defined conditions. The ranking is done on a scale of 1 to 5, where 1 is the highest rank and 5 is the lowest. The output images are organized into folders named according to their rank.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-segregation-model.git
   cd image-segregation-model
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place the images you want to process in the `input` folder.

2. Run the model:
   ```bash
   python main.py
   ```

3. The ranked images will be saved in the `outputs` folder, with each image placed in a subfolder corresponding to its rank (1-5).

## Input and Output Structure

### Input Structure
- `input/`
  - `image1.jpg`
  - `image2.jpg`
  - `image3.jpg`
  - ...

### Output Structure
- `outputs/`
  - `1/`
    - `image1.jpg`
    - `image3.jpg`
  - `2/`
    - `image2.jpg`
  - `3/`
    - `image4.jpg`
  - `4/`
    - `image5.jpg`
  - `5/`
    - `image6.jpg`

## Custom Conditions

To customize the ranking conditions, modify each of the ranking functions in `main.py`. The function should take in an image and return a rank between 1 and 5 based on your specific criteria.

---