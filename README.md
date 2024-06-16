# Emotion Detection 

This project involves building a Convolutional Neural Network (CNN) to classify emotions from facial images.

### Directory Structure

- `src/data_loader.py`: Loading and preprocessing data.
- `src/data_preparation.py`: Script to prepare the data to be loaded
- `src/model.py`: CNN model.
- `src/train.py`: Training loop
- `src/evaluate.py`: Evaluation of model on test data.
- `src/notebooks/`: Jupyter notebook, experiments conducted.
- `data/`: Directory to store the dataset
- `models/`: Best model output
- `requirements.txt`: Libraries to run the project

### Dataset

The dataset should be structured with separate folders for training, validation, and testing images. Each folder should have subfolders named according to the class labels (e.g., `happy`, `sad`).

### Instructions to Run
1. In the terminal run:  `https://github.com/adhilcodes/Emotion-classifier.git`
2. Enter to parent directory:  `cd Emotion-classifier`
3. Install required libraries:  `pip install -r requirements.txt`

4. Prepare the data:  `python src/data_preparation.py`

5. Training the Model:  `python src/train.py`

6. To evaluate the model:  `python src/evaluate.py`

(or)

Jupyter Notebook added in the notebook folder(While running make sure to change the Dataset path)

### My Result:

Test Accuracy: 0.6437
