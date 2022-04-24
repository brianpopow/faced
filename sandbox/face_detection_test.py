import cv2
from time import time
from faced import FaceDetector
from faced import utils
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(message)s')

# for local testing
model_path = "../faced/models/"
# inside the container
# model_path = "/app/networks/"

face_detector = FaceDetector()

img_path = "faces.jpg"
thresh = 0.8
bgr_img = cv2.imread(img_path)
rgb_img = cv2.cvtColor(bgr_img.copy(), cv2.COLOR_BGR2RGB)

# Receives RGB numpy image (HxWxC) and
# returns (x_center, y_center, width, height, prob) tuples.
logger.info('starting face detection ...')
start_time = time()
face_detection_list = face_detector.predict(frame=rgb_img, thresh=thresh)
detected_faces = len(face_detection_list)
end_time = time()
duration = end_time - start_time
logger.info(f'face detection took {duration:.2f} seconds')
logger.info(f'found boxes: {detected_faces}')

increase_box_percentage = 0.5
for i in range(detected_faces):
    logger.info(face_detection_list[i])

ann_img = utils.annotate_image(bgr_img, face_detection_list)
cv2.imwrite("found_boxes.jpg", ann_img)

#blurred_img = image_utils.blur_boxes(image=bgr_img, face_detection_list=face_detection_list, kernel_size=0, sigma=25)
#cv2.imwrite("blurred_boxes.jpg", blurred_img)
