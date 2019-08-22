"""Face detection and emotion classification.

Based on https://github.com/oarriaga/face_classification/tree/master
"""

# model imports
import cv2
import numpy as np
from tensorflow.python.keras.models import load_model

# parameters for loading data and images
DETECTION_MODEL_PATH = 'haarcascade_files/haarcascade_frontalface_default.xml'
EMOTION_MODEL_PATH = 'trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5'

# labels for predictions
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]


class Model:
    """Class for running inference on pre-trained models."""

    def __init__(self):
        """Initialize detection and classification models."""
        self.face_detection = cv2.CascadeClassifier(DETECTION_MODEL_PATH)
        self.emotion_classifier = load_model(EMOTION_MODEL_PATH, compile=False)

        # get input model shape for inference
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]

    def detect_faces(self, image):
        """Detect faces in given image using Haar Cascade classifier.

        Args:
            image: grayscale image as 2D numpy array of shape (height, width)

        Returns:
            List of (x, y, width, height) tuples of all found faces
                (image coordinates), sorted largest to smallest

        """
        faces = self.face_detection.detectMultiScale(
                    image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE)
        return sorted(faces, reverse=True,
                      key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

    def predict_emotions(self, face_images):
        """Predict emotions from given list of face_images.

        Args:
            face_images: list of grayscale images containing
                regions of interest as 2D numpy arrays of shape (height, width)

        Returns:
            Array of probabilities (aligned with EMOTIONS).

        """
        # resize images to the correct size, and then prepare for
        # classification via the CNN
        input_images = [cv2.resize(face_image, self.emotion_target_size)
                        for face_image in face_images]
        input_images = np.asarray(input_images)
        input_images = input_images.astype("float") / 255.0
        input_images = np.expand_dims(input_images, axis=3)

        return self.emotion_classifier.predict(input_images)
