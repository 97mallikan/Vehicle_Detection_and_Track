import cv2
import numpy as np

# Open video file or webcam
# Use 0 for webcam, or replace with "video.mp4"
cap = cv2.VideoCapture("Screencast from 2026-02-26 14-00-50.webm")

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Create KNN background subtractor
backSub = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400.0,
    detectShadows=True
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # Apply background subtraction
    fgMask = backSub.apply(frame)

    # Remove shadow pixels if detectShadows=True
    # Shadows are usually gray (127), foreground is white (255)
    _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_DILATE, kernel, iterations=2)

    # Find contours of moving objects
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore very small regions
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, "Moving Object", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show results
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Foreground Mask", fgMask)
    cv2.imshow("Binary Mask", cleaned)
    #cv2.imshow("Detected Moving Objects", output)

    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or q to quit
        break

cap.release()
cv2.destroyAllWindows()