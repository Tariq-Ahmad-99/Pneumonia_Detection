# Pneumonia Detection using VGG19

## Project Overview
This project is a pneumonia detection system that classifies chest X-ray images into "Normal" and "Pneumonia" using a Convolutional Neural Network (CNN) based on the VGG19 architecture. The model is trained using TensorFlow and deployed via Flask.

## Dataset 
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## File Structure
```
PNEUMONIA_DETECTION/
│-- data/
│   ├── test/   # Test dataset
│   ├── train/  # Training dataset
│   ├── val/    # Validation dataset
│
│-- main/
│   ├── __pycache__/
│   ├── dataAug.py    # Data augmentation script
│   ├── dataLoad.py   # Data loading script
│   ├── test.py       # Testing script
│   ├── train.py      # VGG19 CNN Architecture
│   ├── train2.py     # Incremental unfreezing and fine tuning
│   ├── train3.py     # Unfreezing and fine tuning the entire network (Main training script)
│
│-- model/
│   ├── model2.h5     # Saved model variant
│   ├── model.h5      # Saved model variant
│   ├── unfrozen.h5   # Final trained model
│
│-- venv/             # Virtual environment
│-- web/
│   ├── static/
│   │   ├── css/      # CSS files
│   │   ├── js/       # JavaScript files
│   ├── templates/
│   │   ├── import.html
│   │   ├── index.html
│   ├── uploads/      # Folder for uploaded images
│   ├── app.py        # Flask application
│
│-- .gitignore        # Git ignore file
```

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/Tariq-Ahmad-99/Pneumonia_Detection.git
   cd Pneumonia_Detection
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Ensure that the dataset is structured properly in the `data/` folder.

## Training the Model
Run the following command to train the model:
```sh
python main/train3.py
```
The `train3.py` script implements full unfreezing and fine-tuning of the entire network.

## Running the Flask App
1. Navigate to the `web/` directory:
   ```sh
   cd web
   ```

2. Start the Flask application:
   ```sh
   python app.py
   ```

3. Open `http://127.0.0.1:5000/` in your browser to upload images for prediction.

## Model Predictions
- The model classifies images into two categories:
  - **Normal**
  - **Pneumonia**

## Notes
- The training process includes **data augmentation** to improve generalization.
- The final model (`unfrozen.h5`) is used for inference.
- If you want to modify the architecture, refer to `train.py` and `train2.py` for different training strategies.

## Author
Developed by **Tariq Abu Alsheikh**

## License
MIT License
