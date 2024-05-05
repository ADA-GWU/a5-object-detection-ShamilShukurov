import cv2
import numpy as np
import easyocr
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to load the image
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

#Function to display image

def showImage(image,title):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


# Function to display the image and capture mouse clicks to select corners
def select_corners(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    points = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            points.append([int(event.xdata), int(event.ydata)])
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
            # Proceed automatically after four points have been selected
            if len(points) == 4:
                fig.canvas.mpl_disconnect(cid)
                plt.close()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return np.array(points, dtype="float32")


# Function for perspective transformation
def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]],
        dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


#Function for text detection
def text_detection(image):
    # Convert the transformed image to grayscale for text detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create MSER object
    mser = cv2.MSER_create()

    # Detect regions in gray scale image
    regions, _ = mser.detectRegions(gray_image)

    # Visualize the detected regions
    visualized_image_1 = image.copy()
    visualized_image_1 = np.array(visualized_image_1)

    # Draw rectangles around the detected regions
    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.polylines(visualized_image_1, [hull], True, (0, 255, 0), 1)


    #=================================== Filtering and Merging ===================================

    mask = np.zeros_like(gray_image)

    # Draw the regions on the mask
    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.drawContours(mask, [hull], -1, (255), -1)

    # Optional: Apply morphological operations to merge adjacent regions
    # Using a smaller kernel and fewer iterations
    kernel = np.ones((3,3), np.uint8)  # Smaller kernel
    mask = cv2.dilate(mask, kernel, iterations=1)  # Fewer iterations

    # Find contours which now represent merged regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and aspect ratio
    filtered_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        area = w * h
        # Use less restrictive thresholds
        if 5 < area < 20000 and 0.1 < aspect_ratio < 10:  # Adjust thresholds to be more inclusive
            filtered_regions.append(contour)

    visualized_image_2 = image.copy()
    visualized_image_2 = np.array(visualized_image_2)

    for region in filtered_regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.polylines(visualized_image_2, [hull], True, (0, 255, 0), 1)
    
    return filtered_regions, visualized_image_1,visualized_image_2


# Function for reading text from image
def image_to_text(image, regions):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    reader = easyocr.Reader(['az'])

    recognized_texts = []
    for i in tqdm(range(len(regions))):

        region = regions[i]
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        roi = gray_image[y:y+h, x:x+w]
        roi_pil = Image.fromarray(roi)
        results = reader.readtext(np.array(roi_pil), paragraph=True)  # Using 'paragraph=True' to better handle possibly multiple lines of text

        for _,text in results:

            recognized_texts.append(text)

    return recognized_texts



# Main script
#"C:/Users/shami/Desktop/ADA_S4/Computer_Vision/Assignment5/receiptBravo.jpg"
image_path = r"receiptBravo.jpg"
image = load_image(image_path)
selected_points = select_corners(image)
transformed_image = four_point_transform(image, selected_points)

showImage(transformed_image,"Transformed Receipt")


regions, visualized_image, visualized_image_2 = text_detection(transformed_image)

showImage(visualized_image,"Receipt Image with Detected Text Regions")

showImage(visualized_image_2,"Receipt Image with Detected Text Regions After Merging")


print(f"Number of regions {len(regions)}")

recognized_texts = image_to_text(transformed_image, regions)

print("All recognized texts:", recognized_texts)
