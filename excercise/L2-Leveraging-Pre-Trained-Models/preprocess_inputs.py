import cv2
import numpy as np

def preprocessing(input_image, height, width):
    '''
    since the cv.imread function reads the images as BGR with H(eight),W(idth),C(hannels) order,
    we have to :
    (1) resize-image (based on the model requirement)
    (2) process images as RGB
    (3) re-shape image for as B(atch), C(hannel), width, heightm B=1, C=3
    '''
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image

def pose_estimation(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the pose estimation model
    preprocessed_image = preprocessing(preprocessed_image, 256, 456)

    return preprocessed_image


def text_detection(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related text detection model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the text detection model
    preprocessed_image = preprocessing(preprocessed_image, 768, 1280)

    return preprocessed_image


def car_meta(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related car metadata model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # TODO: Preprocess the image for the car metadata model
    preprocessed_image = preprocessing(preprocessed_image, 72, 72)

    return preprocessed_image
