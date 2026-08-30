import cv2
import numpy as np


def order_points(points):
    """
    Orders four corner points as:
    top-left, top-right, bottom-right, bottom-left.
    """

    points = points.reshape(4, 2)

    ordered = np.zeros((4, 2), dtype=np.float32)

    # Sum of x and y
    s = points.sum(axis=1)

    # Difference between x and y
    diff = np.diff(points, axis=1).reshape(4)

    ordered[0] = points[np.argmin(s)]       # Top-left
    ordered[1] = points[np.argmin(diff)]    # Top-right
    ordered[2] = points[np.argmax(s)]       # Bottom-right
    ordered[3] = points[np.argmax(diff)]    # Bottom-left

    return ordered


def scan_document(image):

    # -----------------------------
    # Step 1: Resize image
    # -----------------------------
    height, width = image.shape[:2]

    scale = 800 / width

    resized = cv2.resize(
        image,
        (800, int(height * scale))
    )

    # -----------------------------
    # Step 2: Convert to grayscale
    # -----------------------------
    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------
    # Step 3: Reduce noise
    # -----------------------------
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # -----------------------------
    # Step 4: Edge detection
    # -----------------------------
    edges = cv2.Canny(
        blur,
        75,
        200
    )

    # -----------------------------
    # Step 5: Find contours
    # -----------------------------
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours by area
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    document = None

    # -----------------------------
    # Step 6: Find document contour
    # -----------------------------
    for contour in contours:

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        # A document should approximately
        # have four corners.
        if len(approx) == 4:

            document = approx
            break

    if document is None:
        return None

    # -----------------------------
    # Step 7: Order corners
    # -----------------------------
    pts = order_points(document)

    tl, tr, br, bl = pts

    # -----------------------------
    # Step 8: Calculate output size
    # -----------------------------
    width1 = np.linalg.norm(br - bl)
    width2 = np.linalg.norm(tr - tl)

    max_width = int(max(width1, width2))

    height1 = np.linalg.norm(tr - br)
    height2 = np.linalg.norm(tl - bl)

    max_height = int(max(height1, height2))

    # -----------------------------
    # Step 9: Destination points
    # -----------------------------
    destination = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # -----------------------------
    # Step 10: Perspective transform
    # -----------------------------
    matrix = cv2.getPerspectiveTransform(
        pts,
        destination
    )

    scanned = cv2.warpPerspective(
        resized,
        matrix,
        (max_width, max_height)
    )

    # -----------------------------
    # Step 11: Enhance output
    # -----------------------------
    scanned_gray = cv2.cvtColor(
        scanned,
        cv2.COLOR_BGR2GRAY
    )

    _, scanned_binary = cv2.threshold(
        scanned_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return scanned_binary


def main():

    # Read input image
    image = cv2.imread(
        "input/sample_document.jpg"
    )

    if image is None:
        print("Error: Image not found.")
        return

    # Scan document
    result = scan_document(image)

    if result is None:
        print("Document could not be detected.")
        return

    # Save result
    cv2.imwrite(
        "output/scanned_document.jpg",
        result
    )

    # Display result
    cv2.imshow(
        "Scanned Document",
        result
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Document scanned successfully!")


if __name__ == "__main__":
    main()
