import unittest
from main import FaceSentimentAnalysis

class TestFaceSentimentAnalysis(unittest.TestCase):
    def test_analyze_sentiment(self):
        # Create an instance of the FaceSentimentAnalysis class
        fsa = FaceSentimentAnalysis()

        # Start the video capture
        fsa.start_video_capture()

        # Analyze the sentiment of the faces in the video
        fsa.analyze_sentiment()

        # Stop the video capture
        fsa.stop_video_capture()

    def test_predict_emotion(self):
        # Create an instance of the FaceSentimentAnalysis class
        fsa = FaceSentimentAnalysis()

        # Load the emotion model
        fsa.load_emotion_model()

        # Make a prediction using the emotion model
        emotion = fsa.predict_emotion(np.zeros((48, 48)))

        self.assertIn(emotion, ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])

    def test_track_emotion(self):
        # Create an instance of the FaceSentimentAnalysis class
        fsa = FaceSentimentAnalysis()

        # Load the emotion model
        fsa.load_emotion_model()

        # Track the emotion of a face over time
        emotion_history = fsa.track_emotion(np.zeros((48, 48)))

        self.assertIsInstance(emotion_history, list)

if __name__ == '__main__':
    unittest.main()