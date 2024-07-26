import os
import csv
import cv2
import torch
import hydra
import easyocr
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

reader = easyocr.Reader(['en'], gpu=True)

def ocr_image(img, coordinates):
    x, y, w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
    img = img[y:h, x:w]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = reader.readtext(gray)
    text = ""

    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
            text = res[1]

    return str(text)

def validate_number(number):
    # Add validation logic as per standard number format
    return number.isalnum() and 6 <= len(number) <= 10

class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch, results_dict):
        p, im, im0 = batch
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{self.dataset.frame}')
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        if len(det) == 0:
            return
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            text_ocr = ocr_image(im0, xyxy)
            if validate_number(text_ocr):
                results_dict[self.data_path.stem] = text_ocr
                break  # Exit after finding the first valid number

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    predictor = DetectionPredictor(cfg)
    results = {}

    class NewFolderHandler(FileSystemEventHandler):
        def on_moved(self, event):
            if event.is_directory and os.path.basename(event.dest_path).startswith('folder_violations'):
                process_new_folder(event.dest_path)

    def process_new_folder(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            img = cv2.imread(image_path)
            predictor.dataset = [(image_path, img, img.copy())]  # single image dataset
            preds = predictor.model.predict(predictor.preprocess(img))
            preds = predictor.postprocess(preds, img, img)
            predictor.write_results(0, preds, predictor.dataset[0], results)
        
        # Write results to CSV
        with open('recognized_plates.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            for id, number in results.items():
                writer.writerow([id, number])
    
    # Initial processing of existing folders
    root_dir = "D:/code/Cropper/Speed/violations4"
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            process_new_folder(folder_path)

    # Setup observer to monitor for moved folders
    event_handler = NewFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, path=root_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    predict()
