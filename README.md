# CarOriNet

## Introduction
`CarOriNet` (Car Orientation Network) is a car image orientation classification project based on PyTorch and torchvision pre-trained models. It supports rapid experimentation with multiple mainstream CNN/ViT models to recognize car image orientations (e.g., front, rear, side, etc.).

## Overall File Structure
```
CarOriNet/
├── config.py              # Model configuration file
├── divide_dataset.py      # Dataset splitting script
├── model_init.py          # Model initialization file
├── models.py              # Orientation classification model definition
├── train.py               # Model training script
├── inference.py           # Model inference script
├── requirements.txt       # Project dependencies
```

### config.py
Used to configure model training parameters, including dataset path, model selection, training hyperparameters, etc.

### divide_dataset.py
Used to split the dataset into training, validation, and test sets (in an 8:1:1 ratio), ensuring class-balanced distribution.

### model_init.py
Used to initialize the model, loading pre-trained models from torchvision and modifying them to adapt to the new classification task.

### models.py
Defines the `OrientationClassifier` class, encapsulating model loading and classification head replacement logic.

### train.py
The core training script of the project, including the full process of data loading, model training, validation, and saving.

### inference.py
Used to load a trained model and perform orientation classification inference on a single image.

### requirements.txt
Install project dependencies using the command `pip install -r requirements.txt`.

## Dataset Preparation and Splitting with divide_dataset.py
Please organize your image dataset in the following structure:
```
dataset/
    class1/
        img1.jpg
        img2.jpg
        ...
    class2/
        img1.jpg
        img2.jpg
        ...
```

Use `divide_dataset.py` to split the dataset:
```bash
python divide_dataset.py --data_dir path_to_your_dataset
```
Parameter description:
- `--data_dir`: Path to the original dataset

After running the above command, the dataset will be split into training, validation, and test sets with the following structure:
```
dataset/
    train/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
    val/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
    test/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
```

## Model Configuration (config.py)
In the `config.py` file, you can configure model training parameters, including but not limited to:
- Dataset path
- Model name (e.g., resnet50, vgg16, vit_b_16, swin_t, etc.)
- Batch size (BATCH_SIZE)
- Number of training epochs (EPOCHS)
- Learning rate (INIT_LR)

## Start Training
Run the following command in the terminal to start training:
```bash
python train.py --config config.py
```
Parameter description:
- `--config`: Configuration file path

## Inference
Use the trained model to perform inference on a single image:
```bash
python inference.py
```

## Console Output and Output Files
During training, the console will output real-time training information, including training loss, validation loss, accuracy, etc. Example output:
```
Training label classes: ['00-Rear', '01-Right Side', '02-Rear Right', '03-Front Right', '04-Front', '05-Left Side', '06-Rear Left', '07-Front Left']

[Train][Epoch 001] Acc: 84.52%
[Val][Epoch 001] Acc: 93.86% | P: 0.938 | R: 0.939 | F1: 0.938
Best model saved (Accuracy: 93.86%)

[Train][Epoch 002] Acc: 94.12%
[Val][Epoch 002] Acc: 95.64% | P: 0.956 | R: 0.956 | F1: 0.956
Best model saved (Accuracy: 95.64%)
...
```

After training, the model will be saved to the specified path, defaulting to the `output` folder.

## Project File Structure After Training
After training, the project file structure may look like this:
```
CarOriNet/
├── data/                          # Dataset
│   ├── test
│   ├── train
│   ├── val
├── output/                        # Output directory
│   ├── swin_t/                    # Model one
│   │   ├── metrics.png            # Training metrics plot
│   │   ├── swin_t_20k_36epochs.pth # Model weights file
├── README.md                      # Project introduction and basic usage
├── config.py                      # Model configuration file
├── divide_dataset.py              # Dataset splitting script
├── model_init.py                  # Model initialization file
├── models.py                      # Orientation classification model definition
├── train.py                       # Model training script
├── inference.py                   # Model inference script
├── requirements.txt               # Project dependencies
```

## License
This project is open-sourced under the MIT License. For details, please refer to the [LICENSE](LICENSE) file.