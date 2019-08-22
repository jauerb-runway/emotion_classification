"""Runway port of face detection and emotion classification.

Based on https://github.com/oarriaga/face_classification/tree/master
"""

import runway
from runway.data_types import image, text, array, vector, image_bounding_box
import emotion_classifier
import utils


@runway.setup
def setup():
    """Initialize and return model."""
    return emotion_classifier.Model()


faces_description = "Bounding boxes of found faces"
detect_description = ("Detect faces in given image and return their "
                      "bounding boxes sorted largest to smallest")


@runway.command('detect', inputs={'image': image},
                outputs={'faces': array(item_type=image_bounding_box,
                                        description=faces_description)},
                description=detect_description)
def detect(model, inputs):
    """Detect faces in a provided image and return their bounding boxes.

    Bounding boxes will be sorted largest to smallest.
    """
    image = inputs['image']

    # convert to gray scale numpy array and detect faces
    gray_image = utils.grayscale_from_pil(image)
    faces = model.detect_faces(gray_image)

    # convert to runway style bounding boxes
    runway_faces = [utils.cv2_to_runway(face, gray_image.shape[1],
                                        gray_image.shape[0]) for face in faces]

    return {'faces': runway_faces}


probabilities_description = ("Probabilities of corresponding face being each "
                             "possible class")
most_likely_description = "Most likely class of corresponding face"
detect_and_classify_description = (
    "Detect faces in given image and return their bounding boxes sorted "
    "largest to smallest along with probabilities of each face's emotion "
    "classification and most likely class.\n Probabilities correspond to:"
    f" {', '.join(emotion_classifier.EMOTIONS)}")


@runway.command('detect_and_classify', inputs={'image': image},
                outputs={'faces': array(item_type=image_bounding_box,
                                        description=faces_description),
                         'probabilities':
                         array(item_type=vector(
                                    length=len(emotion_classifier.EMOTIONS)),
                               description=probabilities_description),
                         'most_likely_classes':
                         array(item_type=text,
                               description=most_likely_description)},
                description=detect_and_classify_description)
def detect_and_classify(model, inputs):
    """Detect and classify faces in given image.

    Returns bounding box of each face sorted largest to smallest along with
    probabilities of each face's emotion classification and most likely class.

    probabilities correspond to emotion_classifier.EMOTIONS
    """
    image = inputs['image']

    # convert to gray scale numpy array and detect faces
    gray_image = utils.grayscale_from_pil(image)
    faces = model.detect_faces(gray_image)

    # extract faces as separate images
    face_images = [utils.extract_roi(gray_image, face) for face in faces]

    # make predictions
    probabilities = model.predict_emotions(face_images)

    # convert to runway style bounding boxes
    runway_faces = [utils.cv2_to_runway(face, gray_image.shape[1],
                                        gray_image.shape[0]) for face in faces]

    most_likely_classes = [emotion_classifier.EMOTIONS[softmax.argmax()]
                           for softmax in probabilities]
    return {'faces': runway_faces, 'probabilities': probabilities,
            'most_likely_classes': most_likely_classes}


classify_description = ("Classify give face image returning probabilities and "
                        "most likely class\n Probabilities correspond to:"
                        f" {', '.join(emotion_classifier.EMOTIONS)}")


@runway.command('classify', inputs={'image': image(
                                description="Cropped face image")},
                outputs={'probabilities':
                         vector(length=len(emotion_classifier.EMOTIONS),
                                description="Probabilities of corresponding "
                                            "face being each possible class"),
                         'most_likely_class':
                             text(description="Most likely class of face")},
                description=classify_description)
def classify(model, inputs):
    """Classify given face image.

    Returns probabilities of face's emotion classification
    and most likely class.

    probabilities correspond to emotion_classifier.EMOTIONS
    """
    image = inputs['image']
    gray_image = utils.grayscale_from_pil(image)
    probabilities = model.predict_emotions([gray_image])[0]
    most_likely_class = emotion_classifier.EMOTIONS[probabilities.argmax()]
    return {'probabilities': probabilities,
            'most_likely_class': most_likely_class}


if __name__ == '__main__':
    runway.run()
