import os
import cv2
import numpy as np
import datetime


class ObjectDetection:
    """
    Класс содержит методы для генерации новой пикчи, содержащей вероятности,
    и расчет того, сколько рублей на фото
    """

    def __init__(self, UPLOAD_FOLDER, CALCULATED_FOLDER):
        self.UPLOAD_FOLDER = UPLOAD_FOLDER
        self.CALCULATED_FOLDER = CALCULATED_FOLDER

        # Load Yolo (default lines for YOLO)
        self.net = cv2.dnn.readNet(
            "yolov4-tiny-custom_final.weights", "yolov4-tiny-custom.cfg")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.classes = []
        with open("obj.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def calculate(self, start_date, alphanumeric_filename):
        self.alphanumeric_filename = alphanumeric_filename
        # Загружаем картинку
        img = cv2.imread(os.path.join(
            self.UPLOAD_FOLDER, self.alphanumeric_filename))

        height, width, channels = img.shape

        # Детектим объекты
        blob = cv2.dnn.blobFromImage(
            img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Показываю инфу на экране
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Объект найден
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Координаты прямоугольника
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        self.generate_image_with_probabilities(
            boxes, indexes, class_ids, confidences, img)

        end_date = datetime.datetime.now()

        execution_time = end_date - start_date

        return {
            "file_name": self.alphanumeric_filename,
            "coins_total": round(self.price_counter(boxes, indexes, class_ids), 2),
            "execution_time": str(
                round((execution_time.microseconds/1000000), 3))}

    def generate_image_with_probabilities(self, boxes, indexes, class_ids, confidences, img):
        """
        Генерим новую картинку с вероятностями
        """

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[class_ids[i]]
                confidence = str(round(confidences[i] * 100, 2))
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)
                cv2.putText(img, label + " (" + confidence + "%)",
                            (x, y - 40), font, 2, color, 3)
        cv2.imwrite(self.CALCULATED_FOLDER + self.alphanumeric_filename, img)

    def price_counter(self, boxes, indexes, class_ids):
        total = 0.0
        for i in range(len(boxes)):
            label = str(self.classes[class_ids[i]])
            if i in indexes:
                if label == "10rub":
                    total += 10
                elif label == "5rub":
                    total += 5
                elif label == "2rub":
                    total += 2
                elif label == "1rub":
                    total += 1
        return total