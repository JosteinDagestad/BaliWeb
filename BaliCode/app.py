from flask_socketio import SocketIO
import cv2
import pathlib
from deepface import DeepFace
import time
from flask import Flask, render_template, Response

app = Flask(__name__)
socketio = SocketIO(app)

# Define the video capture object
video = cv2.VideoCapture(0)

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

face_detected = False  # Flag to track if a face has been detected
current_face_id = None  # Identifier for the currently detected face
last_analysis_time = 0  # Time of the last face analysis
analysis_interval = 3  # Analyze every 3 seconds
analysis_enabled = False  # Flag to control face analysis

# Initialize variables outside the loop
last_detected_gender = "N/A"
last_probability_male = 0.0
last_probability_female = 0.0

button_pressed = False

def draw_text_box(frame, text, y_offset=120):
    # Display detected gender and age in a text box at the bottom of the frame
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]  # Decreased font scale to 0.5
    text_x = int((frame.shape[1] - text_size[0]) / 2)  # Center the text horizontally
    text_y = frame.shape[0] - y_offset  # Position at the bottom of the frame
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)  # Modified font scale

def toggle_analysis():
    # Toggle the analysis flag
    global analysis_enabled
    analysis_enabled = not analysis_enabled

def generate_frames():
    global button_pressed, face_detected, current_face_id, last_detected_gender, last_probability_male, last_probability_female
    last_analysis_time = time.time()  # Initialize it before the loop

    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Initialize button_pressed here
        button_pressed = False

        if analysis_enabled:
            # Extract the face region from the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = clf.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, width, height) in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)

                # Extract the face region from the frame
                face_roi = frame[y:y + height, x:x + width]

                try:
                    # If a new face is detected or the previous face disappears, reset the flag and face ID
                    if not face_detected or current_face_id is None or current_face_id != id(face_roi):
                        face_detected = False
                        current_face_id = id(face_roi)

                        # Analyze the face using DeepFace every 'analysis_interval' seconds
                        current_time = time.time()
                        if current_time - last_analysis_time >= analysis_interval:
                            results = DeepFace.analyze(face_roi, actions=["gender"], enforce_detection=False)

                            # Get the detected gender and probabilities
                            last_detected_gender = results[0]['dominant_gender']
                            last_probability_male = results[0]['gender']['Man']
                            last_probability_female = results[0]['gender']['Woman']

                            # Set the flag to True to indicate that a face has been detected and analyzed
                            face_detected = True

                except ValueError as ve:
                    print(f"Error analyzing face: {ve}")

        # Downsize the frame by 0.5
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Draw the text box on the frame
        text_box = f"Dominant Gender: {last_detected_gender}, Probability Male: {last_probability_male:.2f}, Probability Female: {last_probability_female:.2f}"
        draw_text_box(frame, text_box, y_offset=120)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('toggle_analysis')
def toggle_analysis_socket():
    toggle_analysis()
    socketio.emit('analysis_toggled', analysis_enabled)

if __name__ == '__main__':
    socketio.run(app, debug=True)
