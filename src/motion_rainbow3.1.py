import cv2
import numpy as np

CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 30
CANNY_APERTURE_SIZE = 3
CANNY_L2_GRADIENT = False
BLUR_KERNEL_SIZE = (3, 3)
MORPH_KERNEL_SIZE = (3, 3)
ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
MORPH_SHAPE = cv2.MORPH_RECT
ROYGBIV_PALETTE_BGR = np.array([
    [0, 0, 139],      # dark red
    [0, 0, 180],      # crimson
    [0, 0, 255],      # pure red
    [0, 45, 255],     # red-orange
    [0, 100, 255],    # orange
    [0, 140, 255],    # deep orange
    [0, 165, 255],    # orange
    [0, 200, 255],    # amber
    [0, 215, 255],    # gold
    [0, 255, 255],    # yellow
    [20, 230, 255],   # light yellow
    [100, 150, 255],  # peach
    [130, 100, 255],  # salmon
    [140, 80, 220],   # coral
    [180, 105, 255],  # hot pink
    [200, 130, 255],  # light pink
    [210, 180, 255],  # pale pink
    [147, 20, 255],   # deep pink
    [180, 0, 210],    # magenta-pink
    [130, 0, 255],    # rose
    [0, 0, 0],        # black
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


def fill_regions_with_palette(region_mask: np.ndarray) -> np.ndarray:
    _, labels = cv2.connectedComponents(region_mask.astype(np.uint8))
    output = np.zeros((*region_mask.shape, 3), dtype=np.uint8)

    for label in np.unique(labels):
        if label == 0:
            continue
        color = ROYGBIV_PALETTE_BGR[label % ROYGBIV_PALETTE_BGR.shape[0]]
        output[labels == label] = color

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

        # Close small gaps in the Canny edge map so edges form more complete contours.
        morph_kernel = cv2.getStructuringElement(MORPH_SHAPE, MORPH_KERNEL_SIZE)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morph_kernel)
        closed_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_CLOSE, morph_kernel)

        inverted = cv2.bitwise_not(closed_edges)
        thick_inverted = cv2.erode(inverted, ERODE_KERNEL, iterations=1)

        # Build a mask of region interiors and fill each region with a fixed palette color.
        region_mask = thick_inverted > 250
        colored_regions = fill_regions_with_palette(region_mask)

        cv2.imshow("bloop", colored_regions)

        if cv2.waitKey(4) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
