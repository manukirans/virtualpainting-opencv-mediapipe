import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.signal import savgol_filter

# Hand Detection Class
class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.70, tracking_confidence=0.70):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,
                                         int(self.detection_confidence), int(self.tracking_confidence))
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20] # thumb, index, middle, ring, pinky.
        self.lm_list = []

    # Find Hands in the Frame
    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    # Get Position of Landmarks
    def find_position(self, img, hand_no=0, draw=True):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lm_list

    # Determine Which Fingers Are Up
    def fingers_up(self):
        fingers = []
        if len(self.lm_list) != 0:
            # Thumb
            if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other Fingers
            for id in range(1, 5):
                if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

# Main Drawing Application
def main():
    brush_thickness = 15
    eraser_thickness = 50

    # Load Header Images for Color Selection
    folder_path = "Headers"
    overlay_list = [cv2.imread(f'{folder_path}/{imPath}') for imPath in os.listdir(folder_path)]
    header = overlay_list[0]

    # Default Drawing Color
    draw_color = (169, 169, 169)

    # Capture Video
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Set Width
    cap.set(4, 720)   # Set Height

    # Initialize Hand Detector
    detector = HandDetector(detection_confidence=0.85, tracking_confidence=0.85)
    xp, yp = 0, 0  # Initial pen/eraser position
    drawing = False  
    img_canvas = np.zeros((720, 1280, 3), np.uint8)  # Canvas for drawing: An empty canvas of the same size as the video frame is initialized. This is a black image where drawings will be made.
    points = []  # Store drawing points

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)

        # Hand Detection
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)
        fingers = [] #To store the state of each finger (up or down).

        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1:]  # Tip of the index finger
            x2, y2 = lm_list[12][1:]  # Tip of the middle finger
            fingers = detector.fingers_up()

            # Ensure fingers list has the correct length before accessing its elements
            if len(fingers) >= 2:
                # Selection Mode
                if fingers[1] and fingers[2]:
                    xp, yp = 0, 0  # Reset previous positions
                    drawing = False
                    points = []  # Clear points when not drawing
                    if y1 < 125: 
                        # Check for color selection
                        if 40 < x1 < 150:
                            header = overlay_list[0]
                            draw_color = (169, 169, 169)
                        elif 195 < x1 < 325:
                            header = overlay_list[1]
                            draw_color = (225, 255, 255)
                        elif 360 < x1 < 490:
                            header = overlay_list[2]
                            draw_color = (0, 0, 255)
                        elif 530 < x1 < 660:
                            header = overlay_list[3]
                            draw_color = (0, 255, 0)
                        elif 700 < x1 < 825:
                            header = overlay_list[4]
                            draw_color = (139, 0, 0)
                        elif 865 < x1 < 990:
                            header = overlay_list[5]
                            draw_color = (0, 255, 255)
                        elif 1065 < x1 < 1250:
                            header = overlay_list[6]
                            draw_color = (0, 0, 0)

                        cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)

                # Drawing Mode
                if fingers[1] and not fingers[2]:
                    if not drawing:
                        xp, yp = x1, y1  # Reset the starting position
                    drawing = True

                    cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)  # Draw circle at fingertip
                    points.append((x1, y1))

                    # Savitzky-Golay Smoothing
                    if len(points) > 10:  # Wait for a few points before smoothing
                        window_length = 7  # Adjust this for the level of smoothing
                        poly_order = 5     # Adjust this for the curve's flexibility

                        x, y = zip(*points)
                        x_smooth = savgol_filter(x, window_length, poly_order)
                        y_smooth = savgol_filter(y, window_length, poly_order)

                        points_smooth = list(zip(x_smooth.astype(int), y_smooth.astype(int)))
                        for i in range(1, len(points_smooth)):
                            cv2.line(img, points_smooth[i - 1], points_smooth[i], draw_color, 
                                     brush_thickness if draw_color != (0, 0, 0) else eraser_thickness)
                            cv2.line(img_canvas, points_smooth[i - 1], points_smooth[i], draw_color, 
                                     brush_thickness if draw_color != (0, 0, 0) else eraser_thickness)

                    xp, yp = x1, y1  # Update for the next iteration

        # Combine the Image and the Canvas
        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY) # Function is used for image binarization. It converts a grayscale image into a binary image based on a threshold value.
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV) #Pixels with a value greater than 50 are set to 0 (black) and those with a value less than or equal to 50 are set to 255 (white).
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv) # masks out the areas of the current frame where the drawings are present on the canvas.
        img = cv2.bitwise_or(img, img_canvas) #T operation adds the drawings from img_canvas to the current frame (img).

        # Add the Header
        img[0:125, 0:1280] = header

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
