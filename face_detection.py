import cv2
import face_recognition
import numpy as np
import os
import pickle
import requests
from datetime import datetime
import base64
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from dotenv import load_dotenv

load_dotenv()

SPRINGBOOT_API_URL = os.getenv('SPRINGBOOT_API_URL', 'http://localhost:8080')
TEMP_IMAGE_DIR = os.getenv('IMAGE_SAVE_PATH', './temp_images')

class FaceRecognitionService:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        # Camera settings for better face detection
        self.camera_warmup_time = 2  # seconds
        self.max_capture_attempts = 3
        self.face_detection_model = 'hog'  # 'hog' is faster, 'cnn' is more accurate
    
    def load_known_faces(self):
        """Load known faces from the backend"""
        try:
            response = requests.get(f'{SPRINGBOOT_API_URL}/api/users/faces')
            if response.status_code == 200:
                users = response.json()
                self.known_face_encodings = []
                self.known_face_names = []
                
                for user in users:
                    if user['faceEncoding']:
                        # Decode base64 face encoding
                        face_encoding = np.frombuffer(
                            base64.b64decode(user['faceEncoding']), 
                            dtype=np.float64
                        )
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(user['name'])
                        
                print(f"Loaded {len(self.known_face_encodings)} known faces")
        except Exception as e:
            print(f"Error loading known faces: {e}")
    
    def register_new_face(self, image_path, name, email, phone):
        """Register a new face with user details"""
        try:
            # Check if image file exists
            if not os.path.exists(image_path):
                return {"success": False, "message": "Image file not found"}
            
            # Load image and get face encoding with better parameters
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(
                image, 
                model='large',  # Use more accurate model for registration
                num_jitters=100  # More samples for better encoding
            )
            
            if len(face_encodings) == 0:
                return {"success": False, "message": "No face found in image"}
            
            face_encoding = face_encodings[0]
            
            # Convert to base64 for storage
            face_encoding_b64 = base64.b64encode(face_encoding.tobytes()).decode('utf-8')
            
            # Send to backend
            user_data = {
                "name": name,
                "email": email,
                "phone": phone,
                "faceEncoding": face_encoding_b64
            }
            
            response = requests.post(f'{SPRINGBOOT_API_URL}/api/users/register', 
                                   json=user_data)
            
            if response.status_code == 200:
                # Reload known faces
                self.load_known_faces()
                return {"success": True, "message": "User registered successfully"}
            else:
                return {"success": False, "message": "Registration failed"}
                
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def recognize_face(self, image_path):
        """Recognize face and record entry"""
        try:
            # Check if image file exists
            if not os.path.exists(image_path):
                return {"success": False, "message": "Image file not found"}
            
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Use multiple detection methods for better results
            face_encodings = face_recognition.face_encodings(
                image, 
                model='small',  # Faster for recognition
                num_jitters=10   # Balanced speed vs accuracy
            )
            
            if len(face_encodings) == 0:
                return {"success": False, "message": "No face found in image"}
            
            face_encoding = face_encodings[0]
            
            # Compare with known faces
            if len(self.known_face_encodings) == 0:
                return {"success": False, "message": "No registered faces found. Please register first.", "needs_registration": True}
            
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding,
                tolerance=0.6  # Adjust tolerance (0.6 is default, lower = stricter)
            )
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(matches) > 0 and True in matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    
                    # Record entry
                    entry_data = {
                        "userName": name,
                        "entryTime": datetime.now().isoformat()
                    }
                    
                    response = requests.post(f'{SPRINGBOOT_API_URL}/api/entries/record', 
                                           json=entry_data)
                    
                    if response.status_code == 200:
                        return {"success": True, "message": f"Welcome {name}! Entry recorded."}
                    else:
                        return {"success": False, "message": "Failed to record entry"}
            
            return {"success": False, "message": "Face not recognized. Please register first.", "needs_registration": True}
            
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def capture_image(self):
        """Improved image capture with better camera handling"""
        cap = None
        try:
            # Try different camera indices if 0 doesn't work
            camera_indices = [0, 1, 2]
            
            for camera_index in camera_indices:
                cap = cv2.VideoCapture(camera_index)
                
                if cap.isOpened():
                    print(f"Using camera index: {camera_index}")
                    break
                else:
                    if cap:
                        cap.release()
                    continue
            
            if not cap or not cap.isOpened():
                print("No camera found")
                return None
            
            # Set camera properties for better image quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Camera warmup - capture and discard several frames
            print("Warming up camera...")
            for _ in range(10):  # Capture 10 frames to let camera adjust
                ret, frame = cap.read()
                if not ret:
                    break
                time.sleep(0.1)  # Small delay between frames
            
            # Wait additional time for camera to fully initialize
            time.sleep(self.camera_warmup_time)
            
            # Capture the actual image
            best_frame = None
            max_faces = 0
            
            # Try multiple captures to get the best one
            for attempt in range(self.max_capture_attempts):
                ret, frame = cap.read()
                if ret:
                    # Check if this frame has faces
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(
                        rgb_frame, 
                        model=self.face_detection_model
                    )
                    
                    if len(face_locations) > max_faces:
                        max_faces = len(face_locations)
                        best_frame = frame.copy()
                        
                    if max_faces > 0:
                        break  # Found a face, use this frame
                        
                time.sleep(0.5)  # Wait between attempts
            
            cap.release()
            cv2.destroyAllWindows()
            
            if best_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(TEMP_IMAGE_DIR, f"temp_image_{timestamp}.jpg")
                
                # Save with high quality
                cv2.imwrite(image_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Verify the image was saved and has the right size
                if os.path.exists(image_path):
                    file_size = os.path.getsize(image_path)
                    if file_size > 1000:  # At least 1KB
                        print(f"Image captured successfully: {image_path} ({file_size} bytes)")
                        return image_path
                    else:
                        print(f"Image file too small: {file_size} bytes")
                        os.remove(image_path)
                        return None
                else:
                    print("Failed to save image file")
                    return None
            else:
                print("Failed to capture any frames")
                return None
                
        except Exception as e:
            print(f"Error capturing image: {e}")
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            return None
    
    def test_camera_advanced(self):
        """Advanced camera testing with detailed diagnostics"""
        results = {
            "success": False,
            "message": "",
            "details": {
                "available_cameras": [],
                "selected_camera": None,
                "resolution": None,
                "face_detection_test": False
            }
        }
        
        try:
            # Test multiple camera indices
            available_cameras = []
            for i in range(5):  # Test indices 0-4
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
            
            results["details"]["available_cameras"] = available_cameras
            
            if not available_cameras:
                results["message"] = "No cameras found"
                return results
            
            # Test the first available camera
            camera_index = available_cameras[0]
            cap = cv2.VideoCapture(camera_index)
            
            if cap.isOpened():
                results["details"]["selected_camera"] = camera_index
                
                # Get resolution
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                results["details"]["resolution"] = f"{width}x{height}"
                
                # Test face detection
                time.sleep(1)  # Camera warmup
                ret, frame = cap.read()
                
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    results["details"]["face_detection_test"] = len(face_locations) > 0
                    
                    results["success"] = True
                    results["message"] = f"Camera {camera_index} working. Resolution: {width}x{height}. Faces detected: {len(face_locations)}"
                else:
                    results["message"] = f"Camera {camera_index} opened but failed to capture frame"
                
                cap.release()
            else:
                results["message"] = f"Failed to open camera {camera_index}"
                
        except Exception as e:
            results["message"] = f"Camera test error: {str(e)}"
        
        return results

# Flask API to serve the face recognition service
app = Flask(__name__)
CORS(app)

face_service = FaceRecognitionService()

@app.route('/api/capture-and-recognize', methods=['POST'])
def capture_and_recognize():
    """Capture image and recognize face"""
    try:
        print("Starting face recognition...")
        image_path = face_service.capture_image()
        
        if not image_path:
            return jsonify({"success": False, "message": "Failed to capture image from camera. Please ensure camera is connected and not in use by another application."})
        
        print(f"Image captured: {image_path}")
        result = face_service.recognize_face(image_path)
        
        # Clean up temporary image
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Temporary image removed: {image_path}")
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in capture_and_recognize: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/api/capture-for-registration', methods=['POST'])
def capture_for_registration():
    """Capture image for registration"""
    try:
        print("Starting image capture for registration...")
        image_path = face_service.capture_image()
        
        if not image_path:
            return jsonify({"success": False, "message": "Failed to capture image from camera. Please ensure camera is connected and not in use by another application."})
        
        print(f"Image captured for registration: {image_path}")
        
        # Verify face is detected in the image
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) == 0:
            # Clean up the image if no face detected
            if os.path.exists(image_path):
                os.remove(image_path)
            return jsonify({"success": False, "message": "No face detected in the image. Please ensure you're facing the camera directly and try again."})
        
        # Keep the image path for registration
        return jsonify({"success": True, "imagePath": image_path, "message": "Face captured successfully! Please fill in your details."})
    
    except Exception as e:
        print(f"Error in capture_for_registration: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/api/register-user', methods=['POST'])
def register_user():
    """Register new user with face"""
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        phone = data.get('phone')
        image_path = data.get('imagePath')
        
        if not all([name, email, phone, image_path]):
            return jsonify({"success": False, "message": "Missing required fields"})
        
        result = face_service.register_new_face(image_path, name, email, phone)
        
        # Clean up temporary image
        if os.path.exists(image_path):
            os.remove(image_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"})

@app.route('/api/test-camera', methods=['GET'])
def test_camera():
    """Advanced camera testing"""
    try:
        result = face_service.test_camera_advanced()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": f"Camera test error: {str(e)}"})

@app.route('/api/camera-diagnostics', methods=['GET'])
def camera_diagnostics():
    """Detailed camera diagnostics"""
    try:
        result = face_service.test_camera_advanced()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": f"Diagnostics error: {str(e)}"})

if __name__ == '__main__':
    # Create temp directory if it doesn't exist
    if not os.path.exists('temp_images'):
        os.makedirs('temp_images')
    
    print("Starting Face Recognition Service...")
    print("Make sure your camera is connected and accessible.")
    print("Testing camera on startup...")
    
    # Test camera on startup
    startup_test = face_service.test_camera_advanced()
    print(f"Camera test result: {startup_test}")
    
  app.run(
    host=os.getenv('PYTHON_HOST', '0.0.0.0'),
    port=int(os.getenv('PYTHON_PORT', 5000)),
    debug=True
)
