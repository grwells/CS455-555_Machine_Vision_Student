'''
    From Google's examples for mediapipe
    https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer
'''
import time
import numpy as np
import logging
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


import cv2 as cv

logger = logging.getLogger(__name__)
logging.basicConfig(filename='gesture_rec.log', level=logging.INFO)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

webcam = cv.VideoCapture(0)
key = ord('k')


# mediapipe configuration for gesture recognition
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def get_hands_output(opencv_img, hands_model):
    # pass in an opencv image, hand model from MediaPipe
    # return an annotated opencv image
    
    # mark for pass by ref, performance optimizer
    opencv_img.flags.writeable = False
    # convert to RGB
    opencv_img = cv.cvtColor(opencv_img, cv.COLOR_BGR2RGB)
    # get hand output
    results = hands_model.process(opencv_img)

    # mark image as writeable
    opencv_img.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                opencv_img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    return opencv_img

def draw_hand_landmarks(opencv_img, hand_landmarks):
    # pass in an opencv image, hand landmarks from MediaPipe
    # return an annotated opencv image
    
    # convert to RGB
    opencv_img = cv.cvtColor(opencv_img, cv.COLOR_BGR2RGB)

    mp_drawing.draw_landmarks(
        opencv_img,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )

    return opencv_img

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # callback for detected gesture
    print('gesture recognition result: {}'.format(result))
    
    # process/prepare output image
    #output_image.flags.writeable = True
    # convert back to OpenCV format
    #output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)
    # draw hand landmarks
    if len(result.gestures) > 0:

        '''
        mp_drawing.draw_landmarks(
            output_image,
            result.hand_landmarks,
        )
        # get top result
        gesture = result.gestures[0][0]
        print('\n\tgesture detected:', gesture.category_name)
        numpy_img_out = np.copy(output_image.numpy_view())
        print('gesture detected, image out', dir(numpy_img_out))
        cv.imshow('hand', numpy_img_out)
        key = cv.waitKey(5)
        '''
        logger.info("gesture detected")
        #draw_hand_landmarks()
        #pil_img = Image(output_image)

    else:

        logger.info('no gesture detected')

# customize base options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# initialize hands model
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# initialize recognizer model
with GestureRecognizer.create_from_options(options) as recognizer:
    while key != ord('s'):
        ret, img = webcam.read()
        if not ret:
            logger.error('no image from webcam')
            break

        # convert cv image to mediapipe image object
        # RGB instead of BGR
        hands_annotated_img = get_hands_output(img, hands)
        cv.imshow("MEDIAPIPE HANDS", cv.flip(hands_annotated_img, 1))
        if cv.waitKey(5) & 0xFF == 27:
            break

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        # note: async here, results attached to callback
        recognizer.recognize_async(mp_img, int(time.time()*1000))

