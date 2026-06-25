import cv2
import numpy as np

BLUR_KERNEL_SIZE = (1, 1)
ROYGBIV_PALETTE_BGR = np.array([
    [0,   0, 0],      # black
    [211,   0, 0],  # violet
    [130,   0,  75],  # indigo
    [255,   0,   0],  # blue
    [  0, 255,   0],  # green
    [  0, 255, 255],  # yellow
    [  0, 165, 255],  # orange
    [  0, 165, 255],  # orange
    [  0,   0, 255],  # red
    [  0,   0, 255],  # red
    [  0,   0, 255],  # red
    [  0,   0, 255],  # red
], dtype=np.uint8)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open default webcam (cv2.VideoCapture(0)).")

    prev_blur = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)

        if prev_blur is None:
            prev_blur = np.copy(blur)

        diff = cv2.absdiff(blur, prev_blur)
        prev_blur = np.copy(blur)

        diff_index = np.minimum((diff.astype(np.uint16) * ROYGBIV_PALETTE_BGR.shape[0]) // 256,
                                ROYGBIV_PALETTE_BGR.shape[0] - 1)
        diff_color = ROYGBIV_PALETTE_BGR[diff_index]

        cv2.imshow("Motion Rainbow Diff", diff_color)

        if cv2.waitKey(2) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
