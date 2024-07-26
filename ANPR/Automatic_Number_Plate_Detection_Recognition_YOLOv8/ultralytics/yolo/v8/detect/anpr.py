from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.cfg import get_cfg
from ultralytics.nn.tasks import attempt_load_one_weight

import torch

def run_anpr(source):
    # Create a simple configuration dictionary
    cfg = {
        'model': "path/to/your/best.pt",
        'source': source,
        'imgsz': (640, 640),
        'conf_thres': 0.25,
        'iou_thres': 0.45,
        'max_det': 1000,
        'device': '',
        'view_img': False,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'nosave': False,
        'classes': None,
        'agnostic_nms': False,
        'augment': False,
        'visualize': False,
        'update': False,
        'project': 'runs/detect',
        'name': 'exp',
        'exist_ok': False,
        'line_thickness': 3,
        'hide_labels': False,
        'hide_conf': False,
        'half': False,
        'dnn': False,
    }

    # Initialize
    device = select_device(cfg['device'])
    model = attempt_load_one_weight(cfg['model'], device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(cfg['imgsz'], stride=stride)

    # Dataloader
    dataset = LoadImages(cfg['source'], img_size=imgsz, stride=stride, auto=pt)
    
    # Run inference
    model.warmup(imgsz=(1 if pt else dataset.bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    all_ocr_results = []
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if cfg['half'] else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=cfg['augment'], visualize=cfg['visualize'])

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, cfg['conf_thres'], cfg['iou_thres'], cfg['classes'], cfg['agnostic_nms'], max_det=cfg['max_det'])

        # Process detections
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if cfg['save_crop'] else im0  # for save_crop
            annotator = Annotator(im0, line_width=cfg['line_thickness'], example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    text_ocr = ocr_image(im0, xyxy)
                    all_ocr_results.append(text_ocr)

                    if cfg['save_txt']:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if cfg['save_conf'] else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if cfg['save_img'] or cfg['save_crop'] or cfg['view_img']:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if cfg['hide_labels'] else (names[c] if cfg['hide_conf'] else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, text_ocr, color=colors(c, True))

    return all_ocr_results

if __name__ == "__main__":
    sample_source = "D:/code/Cropper/Speed/violations4/vehicle_34"
    print(run_anpr(sample_source))