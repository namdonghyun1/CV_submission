# yolo_wrapper.py
import os
import yaml
from datetime import datetime
from .train import train, parse_opt
from .detect import run
from ultralytics import YOLO
import torch

def train_model(ex_dict):
    """
    train.py를 사용하여 YOLO 모델을 학습합니다.
    Args:
        ex_dict: 실험 매개변수가 포함된 딕셔너리.
    Returns:
        학습 결과와 모델 경로가 포함된 업데이트된 ex_dict.
    """
    # 학습 시간 및 이름 설정
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"

    # train.py용 옵션 준비
    opt = parse_opt(known=True)
    opt.data = ex_dict['Data Config']
    opt.epochs = ex_dict['Epochs']
    opt.batch_size = ex_dict['Batch Size']
    opt.imgsz = ex_dict['Image Size']
    opt.device = str(ex_dict['Device']).replace("cuda:", "") if "cuda" in str(ex_dict['Device']) else "cpu"
    opt.project = ex_dict['Output Dir']
    opt.name = name
    opt.exist_ok = True
    opt.patience = 20
    opt.weights = ''  # 처음부터 학습
    opt.cfg = f"{ex_dict['Model Name']}.yaml"
    opt.adam = ex_dict['Optimizer'] == 'AdamW'
    opt.hyp = {
        'lr0': ex_dict['LR'],
        'weight_decay': ex_dict['Weight Decay'],
        'momentum': ex_dict['Momentum'],
        'lrf': 0.01,  # 기본 최종 학습률 비율
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 0.05,
        'cls': 0.5,
        'obj': 1.0,
        'label_smoothing': 0.0,
    }

    # 하이퍼파라미터를 임시 파일로 저장
    hyp_path = os.path.join(ex_dict['Output Dir'], 'hyp_temp.yaml')
    with open(hyp_path, 'w') as f:
        yaml.safe_dump(opt.hyp, f, sort_keys=False)
    opt.hyp = hyp_path

    # 학습 실행
    device = torch.device(opt.device if opt.device != 'cpu' else 'cpu')
    train_results = train(opt.hyp, opt, device, callbacks=None)

    # 결과 저장
    pt_path = f"{ex_dict['Output Dir']}/{name}/weights/best.pt"
    ex_dict['PT path'] = pt_path
    ex_dict['Train Results'] = train_results

    # 최적 모델 가중치 로드
    ex_dict['Model'] = YOLO(pt_path)

    # 임시 하이퍼파라미터 파일 삭제
    if os.path.exists(hyp_path):
        os.remove(hyp_path)

    return ex_dict

def detect_and_save_bboxes(model, image_paths):
    """
    detect.py를 사용하여 탐지를 수행합니다.
    Args:
        model: YOLO 모델 인스턴스 또는 가중치 경로.
        image_paths: 처리할 이미지 경로 리스트.
    Returns:
        탐지 결과가 포함된 딕셔너리.
    """
    # detect.py용 옵션 준비
    opt = {
        'weights': model if isinstance(model, str) else model.ckpt_path,
        'source': '',  # 이미지별로 설정
        'imgsz': [640, 640],
        'conf_thres': 0.25,
        'iou_thres': 0.45,
        'max_det': 1000,
        'device': 'cpu',
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'nosave': True,
        'classes': None,
        'agnostic_nms': False,
        'augment': False,
        'visualize': False,
        'update': False,
        'project': 'runs/detect',
        'name': 'exp',
        'exist_ok': True,
        'line_thickness': 3,
        'hide_labels': False,
        'hide_conf': False,
        'half': False,
        'dnn': False,
    }

    results_dict = {}
    for img_path in image_paths:
        opt['source'] = img_path
        # 탐지 실행
        # detect.py는 결과를 직접 반환하지 않으므로 모델을 직접 사용
        results = model(img_path, verbose=False, task='detect')
        img_results = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                img_results.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        results_dict[img_path] = img_results

    return results_dict