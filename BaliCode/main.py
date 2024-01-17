import cv2
import pathlib
from deepface import DeepFace
import time

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

video = cv2.VideoCapture(0)
face_detected = False  # Flag to track if a face has been detected
current_face_id = None  # Identifier for the currently detected face
last_analysis_time = 0  # Time of the last face analysis
analysis_interval = 5  # Analyze every 5 seconds
analysis_enabled = False  # Flag to control face analysis

# Initialize variables outside the loop
last_detected_gender = "N/A"
last_probability_male = 0.0
last_probability_female = 0.0

button_pressed = False

def draw_text_box(frame, text, y_offset=0):
    # Display detected gender and age in a text box at the bottom of the frame
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = int((frame.shape[1] - text_size[0]) / 2)
    text_y = frame.shape[0] - 20 - y_offset
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

def draw_button(frame, button_text, button_pos, button_size):
    # Draw a button on the screen
    cv2.rectangle(frame, button_pos, (button_pos[0] + button_size[0], button_pos[1] + button_size[1]), (0, 255, 0), -1)
    
    # Calculate text position to center it within the button
    text_size = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = int((button_size[0] - text_size[0]) / 2) + button_pos[0]
    text_y = int((button_size[1] - text_size[1]) / 2) + button_pos[1] + int(text_size[1] * 1.5)
    
    cv2.putText(frame, button_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def toggle_analysis():
    # Toggle the analysis flag
    global analysis_enabled
    analysis_enabled = not analysis_enabled

def handle_mouse_event(event, x, y, flags, param):
    global button_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_pos[0] < x < button_pos[0] + button_size[0] and button_pos[1] < y < button_pos[1] + button_size[1]:
            button_pressed = not button_pressed

# Create a window and set the mouse callback
cv2.namedWindow("Faces")
cv2.setMouseCallback("Faces", handle_mouse_event)

while True:
    ret, frame = video.read()

    if not ret:
        break

    if button_pressed:
        toggle_analysis()
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

        for (x, y, width, hight) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + hight), (255, 255, 0), 2)

            # Extract the face region from the frame
            face_roi = frame[y:y + hight, x:x + width]

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

                        # Update the last analysis time
                        last_analysis_time = current_time

            except ValueError as ve:
                print(f"Error analyzing face: {ve}")

    # Calculate the button position at the bottom center
    button_size = (150, 80)
    button_pos = ((frame.shape[1] - button_size[0]) // 2, frame.shape[0] - button_size[1] - 10)
    button_text = "Start/Stop"

    # Draw the text box on the frame
    text_box = f"Dominant Gender: {last_detected_gender}, Probability Male: {last_probability_male:.2f}, Probability Female: {last_probability_female:.2f}"
    draw_text_box(frame, text_box, y_offset=120)

    # Draw the start/stop button
    draw_button(frame, button_text, button_pos, button_size)

    # Display the frame
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
        cv2.imshow("Faces", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()
