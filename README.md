# Sign Language Alphabet Recognition
Run the the corresponding Jupiter Notebooks inside the ``notebooks`` folder in the following order:

## 1 - Installations
The following required Python packages will be installed inside this notebook:
- MediaPipe
- OpenCV
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- tqdm
- Joblib


### **!!! IMPORTANT !!!**
Ensure that Python version ***``3.12.xx``*** is installed, as the Mediapipe package will not install or run correctly with other (newer) versions.


## 2 - MediaPipe
In this notebook, we'll explore how MediaPipe works by testing it on a single image of a woman with both hands visible, and we'll visualize the results.


## 3 - Preprocessing
In this notebook, we preprocess the American Sign Language Alphabet dataset (both images and annotations).

The workflow consists of the following main steps:
1. Map image IDs to their labels using the provided COCO annotation files.
2. Extract hand landmarks from each image using MediaPipe, resulting in 3D coordinate data for each hand.
3. Create structured datasets (train, validation, test) from the extracted landmark data.
4. Preprocess and scale all features to normalize hand size and position.
5. Save the final processed data and the trained scaler into a single .npz file for easy reuse.
6. Visualize random samples using a plotting function to verify that the landmarks and labels are correct.


### **!!! IMPORTANT !!!**
Before running this notebook, make sure you have unzipped
``American Sign Language Letters.v1-v1.coco.zip`` into the ``data/`` directory.

Your folder structure should look like this:

```
data/
└── American Sign Language Letters.v1-v1.coco/
    ├── train/
    │   ├── _annotations.coco.json
    │   ├── A0_jpg.rf.xxxxxxxx.jpg
    │   ├── ...
    ├── test/
    │   ├── _annotations.coco.json
    │   ├── A0_jpg.rf.xxxxxxxx.jpg
    │   ├── ...
    ├── valid/
    │   ├── _annotations.coco.json
    │   ├── A0_jpg.rf.xxxxxxxx.jpg
    │   ├── ...
    └── README.roboflow.txt
```


## 4 - Model
The main objective of this notebook is to train and evaluate multiple machine learning models (four in total) in order to determine which one performs best.
Each model is trained and validated on the preprocessed sign language data. After the initial training, a grid search is performed to tune the hyperparameters and explore different parameter combinations. The model achieving the highest accuracy on both the validation and test sets after tuning is selected as the final model.

Once the best-performing model is identified, it will be saved as a .pkl file for later use.
Afterward, we also load the same scaler used during preprocessing to ensure that all incoming data (e.g., live webcam frames) are processed in exactly the same way as the training data.
Finally, this scaler is saved separately so it can be reused consistently throughout the project.


## 5 - Testing
Now that the best-performing model has been selected and trained, we can move on to testing and real-time usage.
To do this, we first load the trained model along with the scaler that was used during preprocessing, ensuring that all new input data are handled consistently with the training phase.

Next, we define a prediction function that takes the detected hand landmarks as input and returns the corresponding letter from the sign language alphabet.

The final step is where the real testing begins:
we initialize MediaPipe for hand detection and drawing, and set up the webcam stream using OpenCV.
The trained model is then integrated into this live feed; a separate window opens where your hand movements are tracked in real time, and the system displays both the predicted letter and a confidence score indicating how certain the model is about its prediction.
