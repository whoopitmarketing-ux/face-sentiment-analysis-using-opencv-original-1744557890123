import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

class FaceSentimentAnalysis:
    def __init__(self):
        """
        Initialize the FaceSentimentAnalysis class.
        
        Attributes:
        - self.emotions (list): A list of possible emotions.
        - self.emotion_model (object): The emotion model used for sentiment analysis.
        - self.video_capture (object): The video capture object.
        """
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_model = None
        self.video_capture = None

    def start_video_capture(self):
        """
        Start the video capture.
        
        Returns:
        - None
        """
        self.video_capture = cv2.VideoCapture(0)

    def stop_video_capture(self):
        """
        Stop the video capture.
        
        Returns:
        - None
        """
        self.video_capture.release()
        cv2.destroyAllWindows()

    def analyze_sentiment(self):
        """
        Analyze the sentiment of the faces in the video.
        
        Returns:
        - None
        """
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)

            # Analyze the sentiment of each face
            for (x, y, w, h) in faces:
                # Extract the face from the frame
                face = gray[y:y+h, x:x+w]

                # Resize the face to the required size
                face = cv2.resize(face, (48, 48))

                # Make a prediction using the emotion model
                emotion = self.predict_emotion(face)

                # Display the emotion on the frame
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # Display the frame
            cv2.imshow('Sentiment Analysis', frame)

            # Exit on key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def predict_emotion(self, face):
        """
        Predict the emotion of a face.
        
        Args:
        - face (numpy array): The face to predict the emotion for.
        
        Returns:
        - emotion (str): The predicted emotion.
        """
        # Load the emotion model if it has not been loaded
        if self.emotion_model is None:
            self.load_emotion_model()

        # Make a prediction using the emotion model
        emotion = self.emotion_model.predict(face.reshape(1, -1))

        return self.emotions[emotion[0]]

    def load_emotion_model(self):
        """
        Load the emotion model.
        
        Returns:
        - None
        """
        # Check if the emotion model file exists
        if os.path.exists('emotion_model.pkl'):
            # Load the emotion model from the file
            with open('emotion_model.pkl', 'rb') as f:
                self.emotion_model = pickle.load(f)
        else:
            # Train the emotion model if it does not exist
            self.train_emotion_model()

    def train_emotion_model(self):
        """
        Train the emotion model.
        
        Returns:
        - None
        """
        # Load the dataset
        dataset = np.load('dataset.npy')

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(dataset[:, :-1], dataset[:, -1], test_size=0.2, random_state=42)

        # Train the emotion model
        self.emotion_model = LogisticRegression()
        self.emotion_model.fit(X_train, y_train)

        # Save the emotion model to a file
        with open('emotion_model.pkl', 'wb') as f:
            pickle.dump(self.emotion_model, f)

    def track_emotion(self, face):
        """
        Track the emotion of a face over time.
        
        Args:
        - face (numpy array): The face to track the emotion for.
        
        Returns:
        - emotion_history (list): The history of emotions for the face.
        """
        # Initialize the emotion history
        emotion_history = []

        # Track the emotion of the face over time
        while True:
            # Make a prediction using the emotion model
            emotion = self.predict_emotion(face)

            # Add the emotion to the emotion history
            emotion_history.append(emotion)

            # Display the emotion history
            print(emotion_history)

            # Exit on key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return emotion_history

# Create an instance of the FaceSentimentAnalysis class
fsa = FaceSentimentAnalysis()

# Start the video capture
fsa.start_video_capture()

# Analyze the sentiment of the faces in the video
fsa.analyze_sentiment()

# Stop the video capture
fsa.stop_video_capture()