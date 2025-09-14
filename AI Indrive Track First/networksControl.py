from ultralytics import YOLO

# Загружаем модель
car_or_not_car = YOLO('./NeuralNetworks/carornotcar.pt')
dirt_or_not_dirt = YOLO("./NeuralNetworks/dirtdetection.pt")
dent_or_dont_dent = YOLO("./NeuralNetworks/dentordontdent.pt")

def check_car(car_image):
    is_image_has_car = 0
    is_car_dirt = 0
    is_car_dent = 0

    #Есть ли машина на фотографии
    results = car_or_not_car(car_image)
    boxes = results[0].boxes
    classes = boxes.cls.cpu().numpy()
    num_cars = (classes == 2).sum()

    if num_cars > 0:
        is_image_has_car = 1
    elif num_cars < 1:
        print("Не обнаружено машин на фотографии.")
        return False
    
    #Есть ли грязь на машине
    results = dirt_or_not_dirt(car_image)
    boxes = results[0].boxes
    classes = boxes.cls.cpu().numpy()
    num_cars = (classes == 0).sum()

    if num_cars <= 0: is_car_dirt = 1

    #Есть ли повреждения
    result = dirt_or_not_dirt(car_image)
    boxes = results[0].boxes
    if boxes is not None and len(boxes.cls) > 0:
        class_idx = int(boxes.cls[0].cpu().numpy())
        class_name = results[0].names[class_idx]
        if class_name.lower() == "dent": is_car_dent = 1 

    return [is_image_has_car, is_car_dirt, is_car_dent]

