import cv2
import numpy as np

BLUR_KERNEL_SIZE = (11, 11)
DOWNSCALE_FACTOR = 0.25
WATERSHED_INTERVAL = 1
ERODE_ITERATIONS = 10
DILATE_ITERATIONS = 4
# ROYGBIV_PALETTE_BGR = np.array([
#     [0, 0, 139],      # dark red
#     [0, 0, 180],      # crimson
#     [0, 0, 255],      # pure red
#     [0, 45, 255],     # red-orange
#     [0, 100, 255],    # orange
#     [0, 140, 255],    # deep orange
#     [0, 165, 255],    # orange
#     [0, 200, 255],    # amber
#     [0, 215, 255],    # gold
#     [0, 255, 255],    # yellow
#     [20, 230, 255],   # light yellow
#     [100, 150, 255],  # peach
#     [130, 100, 255],  # salmon
#     [140, 80, 220],   # coral
#     [180, 105, 255],  # hot pink
#     [200, 130, 255],  # light pink
#     [210, 180, 255],  # pale pink
#     [147, 20, 255],   # deep pink
#     [180, 0, 210],    # magenta-pink
#     [130, 0, 255],    # rose
#     [0, 0, 0],        # black
# ], dtype=np.uint8)

ROYGBIV_PALETTE_BGR = np.array([
    [0, 0, 10],        # black
    [94, 64, 64],
    [128, 138, 128],
    [192, 192, 222],
    [255, 245, 245],
], dtype=np.uint8)




def compute_watershed_labels(color_frame: np.ndarray, blurred_gray: np.ndarray) -> np.ndarray:
    _, thresh = cv2.threshold(
        blurred_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    sure_bg = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)
    sure_fg = cv2.erode(thresh, None, iterations=ERODE_ITERATIONS)
    unknown = cv2.subtract(sure_bg, sure_fg)

    markers = cv2.connectedComponents(sure_fg)[1]
    markers = markers.astype(np.int32, copy=False)
    markers += 1
    markers[unknown == 255] = 0

    cv2.watershed(color_frame, markers)
    return markers


def label_to_color(label: int) -> np.ndarray:
    return ROYGBIV_PALETTE_BGR[label % ROYGBIV_PALETTE_BGR.shape[0]]


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    output = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    unique_labels = np.unique(label_map)

    for label in unique_labels:
        color = label_to_color(int(label))
        output[label_map == label] = color

    return output


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open the default webcam.")

    label_map = None
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break

        downscaled = cv2.resize(
            frame,
            dsize=None,
            fx=DOWNSCALE_FACTOR,
            fy=DOWNSCALE_FACTOR,
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(downscaled, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)

        if frame_idx % WATERSHED_INTERVAL == 0 or label_map is None:
            label_map = compute_watershed_labels(downscaled, blurred)

        colored_regions = colorize_label_map(label_map)
        upscaled = cv2.resize(
            colored_regions,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        cv2.imshow("Motion Rainbow Watershed Mosaic", upscaled)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
