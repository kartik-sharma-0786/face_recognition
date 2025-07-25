import os
import csv
import time
import datetime
import numpy as np
import cv2
from mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from keras_facenet import FaceNet
import joblib
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Constants
FACE_DETECTION_THRESHOLD = 0.9
RECOGNITION_THRESHOLD = 0.7
MAX_CONSECUTIVE_MISSES = 15  # ~0.5-1 second at 30 FPS
ATTENDANCE_FILE = "attendance.csv"
MODEL_FILE = "face_classifier.joblib"
ENCODER_FILE = "label_encoder.joblib"

class FaceRecognitionSystem:
    def __init__(self):
        self.detector = MTCNN()
        self.embedder = FaceNet()
        self.classifier = None
        self.label_encoder = None
        self.current_attendance = {}  # name: (entry_time, last_seen, misses)
        self.attendance_log = []

    def preprocess_image(self, image):
        """Resize and normalize image for FaceNet"""
        image = cv2.resize(image, (160, 160))
        image = image.astype('float32')
        image = (image - 127.5) / 128.0  # FaceNet-specific normalization
        return np.expand_dims(image, axis=0)

    def extract_faces(self, image):
        """Detect faces in an image using MTCNN"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb_image)
        
        faces = []
        for res in results:
            if res['confidence'] < FACE_DETECTION_THRESHOLD:
                continue
            x, y, w, h = res['box']
            # Ensure coordinates are within image bounds
            x, y = max(0, x), max(0, y)
            face = rgb_image[y:y+h, x:x+w]
            faces.append((face, (x, y, w, h)))
        return faces

    def train_model(self, dataset_path):
        """Train classifier on facial embeddings"""
        X, y = [], []
        
        for person_name in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Warning: Could not read image {image_path}")
                        continue
                    
                    faces = self.extract_faces(img)
                    if not faces:
                        print(f"No face detected in {image_path}")
                        continue
                    
                    # Use first detected face per image
                    face_img, _ = faces[0]
                    preprocessed = self.preprocess_image(face_img)
                    embedding = self.embedder.embeddings(preprocessed)[0]
                    
                    X.append(embedding)
                    y.append(person_name)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
        
        if not X:
            raise ValueError("No training data found. Check your dataset path.")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train classifier
        self.classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        self.classifier.fit(X, y_encoded)
        
        # Save model components
        joblib.dump(self.classifier, MODEL_FILE)
        joblib.dump(self.label_encoder, ENCODER_FILE)
        print(f"Trained model saved to {MODEL_FILE} and {ENCODER_FILE}")

    def load_model(self):
        """Load pre-trained classifier and encoder"""
        if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
            raise FileNotFoundError("Model files not found. Train first.")
        
        self.classifier = joblib.load(MODEL_FILE)
        self.label_encoder = joblib.load(ENCODER_FILE)

    def recognize_faces(self, frame):
        """Recognize faces in a video frame"""
        faces = self.extract_faces(frame)
        current_time = datetime.datetime.now()
        recognized_names = []

        for face_img, (x, y, w, h) in faces:
            try:
                # Process face and get embedding
                preprocessed = self.preprocess_image(face_img)
                embedding = self.embedder.embeddings(preprocessed)[0]
                
                # Predict identity
                predictions = self.classifier.predict_proba([embedding])[0]
                max_index = np.argmax(predictions)
                confidence = predictions[max_index]
                
                if confidence > RECOGNITION_THRESHOLD:
                    name = self.label_encoder.inverse_transform([max_index])[0]
                    recognized_names.append(name)
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except Exception as e:
                print(f"Recognition error: {str(e)}")
        
        return recognized_names, current_time, frame

    def update_attendance(self, recognized_names, current_time):
        """Update attendance records based on recognized faces"""
        # Reset miss counter for recognized faces
        for name in recognized_names:
            if name in self.current_attendance:
                entry_time, _, _ = self.current_attendance[name]
                self.current_attendance[name] = (entry_time, current_time, 0)
            else:
                # New entry
                self.current_attendance[name] = (current_time, current_time, 0)
                print(f"NEW ENTRY: {name} at {current_time}")

        # Update miss counters and check for exits
        exited = []
        for name, (entry_time, last_seen, misses) in list(self.current_attendance.items()):
            if name not in recognized_names:
                misses += 1
                if misses >= MAX_CONSECUTIVE_MISSES:
                    duration = (last_seen - entry_time).total_seconds()
                    self.attendance_log.append({
                        'name': name,
                        'date': entry_time.date(),
                        'entry_time': entry_time.time(),
                        'exit_time': last_seen.time(),
                        'duration': duration
                    })
                    exited.append(name)
                else:
                    self.current_attendance[name] = (entry_time, last_seen, misses)
        
        # Remove exited persons
        for name in exited:
            del self.current_attendance[name]
            print(f"EXIT: {name} after {self.attendance_log[-1]['duration']:.1f} seconds")

    def save_attendance(self):
        """Save attendance records to CSV file"""
        try:
            file_exists = os.path.isfile(ATTENDANCE_FILE)
            with open(ATTENDANCE_FILE, 'a', newline='') as f:
                fieldnames = ['name', 'date', 'entry_time', 'exit_time', 'duration']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                for record in self.attendance_log:
                    writer.writerow(record)
            
            print(f"Attendance records saved to {ATTENDANCE_FILE}")
            self.attendance_log = []
        except PermissionError:
            print(f"Error: Permission denied when writing to {ATTENDANCE_FILE}")
        except Exception as e:
            print(f"Error saving attendance: {str(e)}")

    def run_attendance_tracker(self, video_source=0):
        """Main function to run attendance tracking"""
        if not self.classifier or not self.label_encoder:
            raise RuntimeError("Model not loaded. Train or load a model first.")
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError("Cannot open video source")
        
        last_save_time = time.time()
        print("Starting attendance tracking. Press 'q' to exit...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Video feed ended")
                    break
                
                # Recognize faces and update attendance
                recognized_names, current_time, frame = self.recognize_faces(frame)
                self.update_attendance(recognized_names, current_time)
                
                # Display frame
                cv2.imshow('Attendance Tracker', frame)
                
                # Auto-save every 30 seconds
                if time.time() - last_save_time > 30:
                    self.save_attendance()
                    last_save_time = time.time()
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Save remaining attendance records
            self.save_attendance()
            
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            print("System shutdown")

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Facial Recognition Attendance System')
    parser.add_argument('--train', help='Path to training dataset directory')
    parser.add_argument('--run', action='store_true', help='Run attendance tracker')
    args = parser.parse_args()

    system = FaceRecognitionSystem()
    
    if args.train:
        print(f"Training model with data from {args.train}")
        system.train_model(args.train)
    elif args.run:
        try:
            system.load_model()
            system.run_attendance_tracker()
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Please specify either --train or --run option")