import cv2 as cv
import numpy as np

def process_pipeline(image_path, xyxy, delta=0.1, limit=5, max_value=255, adaptive_method=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                     threshold_type=cv.THRESH_BINARY_INV, block_size=15, C=7, kernel_size=(2, 2), iterations=1):
    im = cv.imread(image_path)
    cropped = crop_bbox(im, xyxy)
    gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

    thresh = cv.adaptiveThreshold(gray, max_value, adaptive_method, threshold_type, block_size, C)

    kernel = np.ones(kernel_size, np.uint8)
    thresh = cv.dilate(thresh, kernel, iterations=iterations)
    thresh = cv.erode(thresh, kernel, iterations=iterations)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    cleaned = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    angles = np.arange(-limit, limit + delta, delta)
    img_stack = np.stack([rotate_image(cleaned, angle) for angle in angles], axis=0)
    scores = determine_score(img_stack)
    best_angle = angles[np.argmax(scores)]
    corrected = rotate_image(cropped, best_angle)
    clean_im = clean_image(corrected, max_value, adaptive_method, threshold_type, block_size, C, kernel_size, iterations)

    return best_angle, clean_im

def crop_bbox(image, xyxy):
    image = image[int(xyxy[0][1]):int(xyxy[0][3]), int(xyxy[0][0]):int(xyxy[0][2])]
    return image

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    corrected = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return corrected

def determine_score(arr):
    histogram = np.sum(arr, axis=2, dtype=float)
    score = np.sum((histogram[..., 1:] - histogram[..., :-1]) ** 2, axis=1, dtype=float)
    return score

def clean_image(im, max_value=255, adaptive_method=cv.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type=cv.THRESH_BINARY_INV, block_size=15, C=7, kernel_size=(2, 2), iterations=1):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    
    blured = cv.GaussianBlur(gray, (5, 5), 0)
    bfilter = cv.bilateralFilter(blured, 11, 17, 17) 
    edged = cv.Canny(bfilter, 30, 200) 

    return edged

def save_temp_image(image, filename='temp.png'):
    cv.imwrite(filename, image)
    return filename