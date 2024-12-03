from ultralytics import YOLO

model = YOLO("./fine_tune/bounding_box.pt")

def getBoundingBox(image_path, confidence=0.8):
    result = model.predict(image_path)
    coordiantes = []
    
    for res in result:
        for box in res.boxes:
            if(box.conf[0] < confidence): continue
            xyxy = box.xyxy[0]
            xyxy = xyxy.cpu().numpy()
            coordiantes.append(xyxy)
        
    return coordiantes