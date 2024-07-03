from flask import Flask, Response, render_template
import cv2
import numpy as np
import time
from config import *

app = Flask(__name__)

# Video capture initialization
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Parameters for virtual table
table_width = TABLE_WIDTH
table_height = TABLE_HEIGHT
camera_height = CAMERA_DISTANCE_TO_TABLE

# Function to adjust saturation and contrast
def adjust_saturation_contrast(frame, saturation_scale, contrast_scale):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = cv2.multiply(hsv[..., 1], saturation_scale)
    adjusted_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    adjusted_frame = cv2.convertScaleAbs(adjusted_frame, alpha=contrast_scale, beta=0)
    return adjusted_frame

# Function to find the largest circle in the mask
def find_largest_circle(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    largest_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    return int(x), int(y), int(radius)

# Function to draw circle points and save coordinates
def draw_circle_points(frame, x, y, radius, num_points=10):
    points = []
    for i in range(num_points):
        angle = i * (360 / num_points)
        cx = int(x + radius * np.cos(np.radians(angle)))
        cy = int(y + radius * np.sin(np.radians(angle)))
        points.append((cx, cy))
        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
        cv2.putText(frame, str(i), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f'{str(i)} ({cx}, {cy})', (10, 80 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame, points

# Function to calculate position and distance on a virtual table
def calculate_position_distance(x, y, table_width, table_height, camera_height):
    table_x = x
    table_y = table_height - y
    return table_x, table_y

# Function to determine if a goal is hit
def check_goal(ball_position, table_width, table_height):
    goal_depth = GOAL_DEPTH
    goal_area_height = GOAL_AREA_HEIGHT

    if ball_position[0] <= goal_depth and \
       (table_height - goal_area_height) // 2 <= ball_position[1] <= (table_height + goal_area_height) // 2:
        return "Left Goal"
    
    if ball_position[0] >= table_width - goal_depth and \
       (table_height - goal_area_height) // 2 <= ball_position[1] <= (table_height + goal_area_height) // 2:
        return "Right Goal"
    
    return None

# Function to draw virtual table with dimensions
def draw_virtual_table(frame, table_width, table_height, ball_position=None):
    table = np.zeros((table_height, table_width, 3), dtype=np.uint8)
    cv2.rectangle(table, (0, 0), (table_width-1, table_height-1), (0, 255, 0), 2)
    cv2.line(table, (table_width // 2, 0), (table_width // 2, table_height-1), (255, 255, 255), 1)
    goal_depth = GOAL_DEPTH
    goal_area_height = GOAL_AREA_HEIGHT
    cv2.rectangle(table, (0, (table_height - goal_area_height) // 2), 
                  (goal_depth, (table_height + goal_area_height) // 2), (0, 0, 255), -1)
    cv2.rectangle(table, ((table_width - goal_depth), (table_height - goal_area_height) // 2), 
                  (table_width, (table_height + goal_area_height) // 2), (0, 0, 255), -1)
    
    if ball_position:
        cv2.circle(table, ball_position, 10, (0, 255, 255), -1)
        cv2.putText(table, f'Ball: {ball_position}', (ball_position[0] + 10, ball_position[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    player1_position = (50, table_height // 2)
    player2_position = (table_width - 50, table_height // 2)
    cv2.circle(table, player1_position, 8, (255, 0, 0), -1)
    cv2.circle(table, player2_position, 8, (0, 0, 255), -1)
    
    return table

# Function to process frames and detect the ball
def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array(LOWER_COLOR)
    upper_yellow = np.array(UPPER_COLOR)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    x, y, radius = find_largest_circle(mask)
    if x is not None and y is not None and radius is not None:
        frame, points = draw_circle_points(frame, x, y, radius)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(frame, f'Center: ({x}, {y})', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame, x, y, radius
    return frame, None, None, None

# Function to resize virtual table to match processed frame height
def resize_virtual_table(virtual_table, height):
    return cv2.resize(virtual_table, (virtual_table.shape[1], height))

# Function to resize processed frame to match virtual table height
def resize_processed_frame(processed_frame, height):
    return cv2.resize(processed_frame, (processed_frame.shape[1], height))

# Generator function for camera stream
def gen_camera():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Processing frame here (adjusting saturation and contrast, detecting ball, etc.)
        processed_frame = adjust_saturation_contrast(frame, saturation_scale=SATURATION_SCALE, contrast_scale=CONTRAST_SCALE)
        processed_frame, x, y, radius = process_frame(processed_frame)
        
        # Generate virtual table visualization
        ball_position = None
        if x is not None and y is not None:
            ball_position = calculate_position_distance(x, y, table_width, table_height, camera_height)
            goal_hit = check_goal(ball_position, table_width, table_height)
            if goal_hit:
                print(f"{goal_hit} was hit!")
        
        virtual_table = draw_virtual_table(np.zeros((table_height, table_width, 3), dtype=np.uint8), table_width, table_height, ball_position)
        virtual_table_resized = resize_virtual_table(virtual_table, processed_frame.shape[0])  # Resize virtual table
                
        # Convert to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', virtual_table_resized)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
