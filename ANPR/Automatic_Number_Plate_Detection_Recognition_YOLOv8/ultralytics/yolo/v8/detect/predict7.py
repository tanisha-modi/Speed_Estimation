# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import easyocr
import csv
import pytesseract
import sys
import os

sys.path.append(os.path.abspath(r'D:\code\Cropper\ESRGAN'))
from test2 import process_image

import cv2
reader = easyocr.Reader(['en'], gpu=False)

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


def ocr_image(img,coordinates, img_name):
    x,y,w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]),int(coordinates[3])
    img = img[y:h,x:w]

    # gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
    # #gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    # result = reader.readtext(gray)
    # text = ""

    img = process_image(img)

    cv2.imwrite(f'D:/code/Cropper/preprocess_img/crop_plate {img_name}.jpg', img)
    gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
    inverted = cv2.bilateralFilter(gray, 15, 17, 17)
    # inverted = cv2.bitwise_not(gray)
    result  = (pytesseract.image_to_string(inverted)).strip()
    result = result.replace('(', '').replace(')', '').replace(',', '').replace('~', '').replace('-', '')
    print(result)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # noise_removed = noise_removal(gray)
    # _ , im_bw = cv2.threshold(inverted, 180, 250, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _ , im_bw = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # im_bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    # thickened = thick_font(inverted)
    # result = reader.readtext(inverted)
    # text = ""

    cv2.imwrite(f'D:/code/Cropper/preprocess_img/output_image {img_name}.jpg', inverted)

    # cv2.imwrite('output_image.jpg', gray)
    # for res in result:
    #     if len(result) == 1:
    #         text = res[1]
    #     if len(result) >1 and len(res[1])>6 and res[2]> 0.2:
    #         text = res[1]
    # #     text += res[1] + " "
    
    # return str(text)
    return result 

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

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
           return "0"
        
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                # print("inside")
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                
                text_ocr = ocr_image(im0,xyxy, f'image_{self.seen}_{idx}')
                label = text_ocr    
                if len(label) == 0:
                    label = "0"
                # print("1245",label)
                # print("123",label)

                # print("hihih",label)         
                # self.annotator.box_label(xyxy, label, color=colors(c, True))
            # if self.args.save_crop:
            #     imc = im0.copy()
            #     save_one_box(xyxy,
            #                  imc,
            #                  file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
            #                  BGR=True)





            # ---------------------------------------------
            # validate krne ka aur csv me likhwane ka code yahi likhna pdega, me value return nhi krwa skti 
            # ---------------------------------------------
        return label


def append_to_csv(file_name, data):
    if isinstance(data, tuple):
        data = [list(data)]
    elif isinstance(data, set):
        data = [list(row) for row in data]
    with open(file_name, 'a', newline='') as file:
        print("hello")
        writer = csv.writer(file)
        writer.writerow(data)  # Write the new data
        print("write successful")

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    # cfg.model = cfg.model or "yolov8n.pt"
    cfg.model=r"D:\code\Cropper\ANPR\Automatic_Number_Plate_Detection_Recognition_YOLOv8\ultralytics\yolo\v8\detect\best.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    # cfg.source=r"D:\code\Cropper\Speed\final\violations4\vehicle_2"
    # cfg.source=r"D:\code\Cropper\cars\car1"
    path = str(cfg.source)
    predictor = DetectionPredictor(cfg)
    number_plate = predictor()
    print(number_plate)
    # append_to_csv('D:/code/Cropper/Speed/final/violation_report.csv', [path, number_plate])
    append_to_csv('D:/code/Cropper/Speed/video/v5/violation_report.csv', [path, number_plate])
    # return ("GIJIJ",predictor())


if __name__ == "__main__":
    print("welcome to number recognition script")
    predict()
    # print(a)

