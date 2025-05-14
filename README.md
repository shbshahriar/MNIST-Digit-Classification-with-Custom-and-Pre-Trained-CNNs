# MNIST Digit Classification with Custom and Pre-Trained CNNs

This project implements various Convolutional Neural Network (CNN) architectures to classify handwritten digits from the MNIST dataset. The project includes custom CNN models as well as pre-trained models like ResNet50 and VGG16.

## Project Structure

```
MNIST Digit Classification with Custom and Pre-Trained CNNs/
├── Code/
│   ├── MNIST_CUSTOM_CNN.ipynb       # Jupyter Notebook for training and evaluating the custom CNN model
│   ├── MNIST_RESNET50.ipynb         # Jupyter Notebook for training and evaluating the ResNet50 model
│   ├── MNIST_VGG16.ipynb            # Jupyter Notebook for training and evaluating the VGG16 model
├── Logs/
│   ├── custom_cnn_training_log.csv  # Training log for the custom CNN model
│   ├── resnet50_training_log.csv    # Training log for the ResNet50 model
│   ├── vgg19_training_log.csv       # Training log for the VGG19 model
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies

```

## Requirements

- Python 3.8 or later
- PyTorch
- torchvision
- Jupyter Notebook
- numpy
- pandas
- matplotlib

## Installation

1. Clone the repository:
   ```powershell
   git clone https://github.com/shbshahriar/MNIST-Digit-Classification-with-Custom-and-Pre-Trained-CNNs.git
   ```

2. Navigate to the project directory:
   ```powershell
   cd MNIST-Digit-Classification-with-Custom-and-Pre-Trained-CNNs
   ```

3. Install the required Python packages:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

1. Open any of the Jupyter Notebooks to train or evaluate a model:
   ```powershell
   jupyter notebook
   ```

2. Navigate to the `Code/` directory and select the desired notebook (e.g., `MNIST_CUSTOM_CNN.ipynb`) and run the cells sequentially.

3. Training logs and saved models will be stored in the `Logs/` and `Model/` directories, respectively.

## Models

### Custom CNN
- A custom-built Convolutional Neural Network designed for digit classification.

### ResNet50
- A pre-trained ResNet50 model fine-tuned on the MNIST dataset.

### VGG16
- A pre-trained VGG16 model fine-tuned on the MNIST dataset.

## Logs

Training logs for each model are stored in the `Logs/` directory. These logs include metrics such as training loss and accuracy over epochs.

## Saved Models

The `Model/` directory contains the saved weights for each trained model.

## Dataset

The MNIST dataset is used for training and evaluation. It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

## License

This project is released without any specific license. It is shared openly for educational and non-commercial use. You are free to use, modify, and share it — no strings attached.

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
