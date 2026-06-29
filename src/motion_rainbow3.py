import cv2
import numpy as np

CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 30
CANNY_APERTURE_SIZE = 3
CANNY_L2_GRADIENT = False
BLUR_KERNEL_SIZE = (11, 11)
ROYGBIV_PALETTE_BGR = np.array([
    [0, 0, 10],
    [94, 64, 64],
    [128, 138, 128],
    [192, 192, 222],
    [255, 245, 245],
], dtype=np.uint8)


def colorize_labels(label_map: np.ndarray) -> np.ndarray:
    output = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    unique_labels = np.unique(label_map)
    for label in unique_labels:
        if label == 0:
            continue
        color = ROYGBIV_PALETTE_BGR[label % ROYGBIV_PALETTE_BGR.shape[0]]
        output[label_map == label] = color
    return output


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open the default webcam.")

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)
        edges = cv2.Canny(
            blurred,
            CANNY_THRESHOLD1,
            CANNY_THRESHOLD2,
            apertureSize=CANNY_APERTURE_SIZE,
            L2gradient=CANNY_L2_GRADIENT,
        )

        inverted = cv2.bitwise_not(edges)

        # num_labels, labels = cv2.connectedComponents(inverted)
        # colored_edges = colorize_labels(labels)

        cv2.imshow("Live Canny Connected Components", inverted)

        if cv2.waitKey(4) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
