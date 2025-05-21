import cv2

def preprocess_image(img_path, size=(128, 128)):
    image = cv2.imread(img_path)
    image = cv2.resize(image, size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray
