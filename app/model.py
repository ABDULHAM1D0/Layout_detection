from ultralytics import YOLO


class LayoutDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image_path: str):
        results = self.model(image_path)
        annotated_img = results[0].plot()
        return annotated_img, results[0]
