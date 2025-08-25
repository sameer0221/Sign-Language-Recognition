
import os
import cv2
import numpy as np


os.makedirs(SAVE_DIR, exist_ok=True)

background = None
accumulated_weight = 0.5

ROI_top, ROI_bottom = 100, 300
ROI_right, ROI_left = 150, 350

def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def find_external_contours(img):
    # Compatible with OpenCV 3 and 4
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    return contours

def segment_hand(frame, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours = find_external_contours(thresholded.copy())
    if len(contours) == 0:
        return None
    hand_segment_max_cont = max(contours, key=cv2.contourArea)
    return (thresholded, hand_segment_max_cont)

cam = cv2.VideoCapture(0)
num_frames = 0
num_imgs_taken = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Could not read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 60:
        cal_accum_avg(gray_frame, accumulated_weight)
        cv2.putText(frame_copy, "Fetching background... please wait",
                    (60, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    else:
        hand = segment_hand(gray_frame)
        if hand is not None:
            thresholded, hand_segment = hand
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
            cv2.imshow("Thresholded Hand", thresholded)

            if num_imgs_taken < NUM_IMAGES_TO_SAVE:
                out_path = os.path.join(SAVE_DIR, f"{num_imgs_taken:04d}.jpg")
                cv2.imwrite(out_path, thresholded)
                num_imgs_taken += 1
                cv2.putText(frame_copy, f"Saved: {num_imgs_taken}/{NUM_IMAGES_TO_SAVE} (Label {LABEL})",
                            (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                print("Done collecting images.")
                break
        else:
            cv2.putText(frame_copy, "No hand detected", (40, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 2)
    cv2.imshow("Sign Detection", frame_copy)
    num_frames += 1

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cam.release()
cv2.destroyAllWindows()
