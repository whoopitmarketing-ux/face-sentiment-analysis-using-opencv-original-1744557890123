import cv2
import numpy as np

def extract_faces(frame):
    """
    Extract faces from a frame.
    
    Args:
    - frame (numpy array): The frame to extract faces from.
    
    Returns:
    - faces (list): A list of faces extracted from the frame.
    """
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)

    return faces

def resize_face(face):
    """
    Resize a face to the required size.
    
    Args:
    - face (numpy array): The face to resize.
    
    Returns:
    - resized_face (numpy array): The resized face.
    """
    # Resize the face to the required size
    resized_face = cv2.resize(face, (48, 48))

    return resized_face