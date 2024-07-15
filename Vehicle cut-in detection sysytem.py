import cv2
import numpy as np

# Load the Haar cascade for vehicle detection
car_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_car.xml')

# Function to detect vehicles
def detect_vehicles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vehicles = car_cascade.detectMultiScale(gray, 1.1, 1)
    return vehicles

# Function to draw bounding boxes around detected vehicles
def draw_boxes(frame, vehicles):
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Function to check for cut-ins
def check_cut_in(previous_positions, current_positions):
    cut_in_detected = False
    for prev in previous_positions:
        for curr in current_positions:
            # Check if the current vehicle is cutting in front of the previous vehicle
            if curr[0] < prev[0] + prev[2] and curr[0] + curr[2] > prev[0] and curr[1] < prev[1] + prev[3] and curr[1] + curr[3] > prev[1]:
                cut_in_detected = True
                break
    return cut_in_detected

# Initialize video capture
cap = cv2.VideoCapture('test_video.mp4')  # Replace with 0 for webcam

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

previous_positions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect vehicles in the current frame
    current_positions = detect_vehicles(frame)

    # Draw bounding boxes around detected vehicles
    draw_boxes(frame, current_positions)

    # Check for cut-ins
    if previous_positions:
        cut_in_detected = check_cut_in(previous_positions, current_positions)
        if cut_in_detected:
            cv2.putText(frame, 'Cut-in Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Vehicle Cut-in Detection', frame)

    # Update previous positions
    previous_positions = current_positions

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
