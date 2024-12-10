from detect import getBoundingBox
from preprocess_image import process_pipeline
from model import apply_ocr

image_path = 'dataset//test//images//AQUA7_56330_checkin_2020-10-27-10-253UBQywQkj_jpg.rf.509cd043dc5fdb5cd2833f598029a7d6.jpg'

coordinates = getBoundingBox(image_path)
_, processed_image = process_pipeline(image_path, coordinates)
result = apply_ocr(processed_image)

print(result)