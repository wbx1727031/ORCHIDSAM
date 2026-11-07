import argparse
import os
import shutil

import cv2
import numpy as np
import paddle
import pandas as pd
from PIL import Image
from paddleseg.models import CCNet, DMNet, ResNet50_vd, UPerNetViTAdapter, ViTAdapter
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_score, recall_score)


def compute_metrics(pred, label, NUM):
    label = np.uint8(label.flatten())
    pred = np.uint8(pred.flatten())
    cm = confusion_matrix(label, pred, labels=range(NUM))
    intersection = np.diag(cm)
    union = cm.sum(1) + cm.sum(0) - intersection
    iou = intersection / (union + 1e-10)
    present = (cm.sum(1) + cm.sum(0)) > 0
    valid_iou = iou[present]
    mIoU = np.mean(valid_iou) if len(valid_iou) > 0 else 0
    f1 = f1_score(label, pred, average='macro', zero_division=0)
    acc = accuracy_score(label, pred)
    prec = precision_score(label, pred, average='macro', zero_division=0)
    rec = recall_score(label, pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(label, pred)
    return mIoU, f1, kappa, acc, prec, rec


def get_image_list(image_path):
    valid_suffix = ['.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png']
    image_list = []
    image_path = os.path.normpath(image_path)
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1].lower() in valid_suffix:
            image_list.append(image_path)
    elif os.path.isdir(image_path):
        for root, _, files in os.walk(image_path):
            for f in files:
                if os.path.splitext(f)[-1].lower() in valid_suffix:
                    image_list.append(os.path.join(root, f))
    return image_list


# Preprocessing must match training: BGR input (no RGB conversion), 512x512 resize.
TARGET_SIZE = (512, 512)


def _read_image_bgr(img_path):
    image_array = np.fromfile(img_path, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot decode image: {img_path}")
    return image


def run_prediction(model, METHOD, img_input_path, output_dir, model_path):
    save_dir = os.path.normpath(output_dir)

    if os.path.exists(save_dir):
        try:
            shutil.rmtree(save_dir)
        except:
            pass
    os.makedirs(os.path.join(save_dir, 'gray'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'overlay'), exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weight file not found: {model_path}")
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    model.eval()

    colormap = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 255], [255, 255, 0], [0, 255, 0]], dtype=np.uint8)
    image_list = get_image_list(img_input_path)

    print(f"\n>>> Prediction | model: {METHOD} | images: {len(image_list)} | output: {save_dir}")

    for img_path in image_list:
        img_name = os.path.basename(img_path)
        img_bgr_raw = _read_image_bgr(img_path)
        orig_h, orig_w = img_bgr_raw.shape[:2]

        img_bgr_resized = cv2.resize(img_bgr_raw, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        img_tensor = paddle.to_tensor(img_bgr_resized.astype('float32') / 255.0).transpose((2, 0, 1)).unsqueeze(0)

        with paddle.no_grad():
            pred = model(img_tensor)
            if isinstance(pred, (list, tuple)): pred = pred[0]
            pred_label_small = paddle.argmax(pred, axis=1).squeeze().numpy().astype(np.uint8)
        pred_label = cv2.resize(pred_label_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        gray_path = os.path.join(save_dir, 'gray', os.path.splitext(img_name)[0] + '_gray.png')
        cv2.imencode('.png', pred_label)[1].tofile(gray_path)
        img_rgb_raw = cv2.cvtColor(img_bgr_raw, cv2.COLOR_BGR2RGB)
        overlay = (img_rgb_raw * 0.5 + colormap[pred_label] * 0.5).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        overlay_path = os.path.join(save_dir, 'overlay', os.path.splitext(img_name)[0] + '_overlay.png')
        cv2.imencode('.png', overlay_bgr)[1].tofile(overlay_path)
        print(f"Done: {img_name}")

    print(f">>> Predictions saved to: {save_dir}")
    return save_dir


def run_evaluation(METHOD, label_dir, output_dir):
    save_dir = os.path.normpath(output_dir)
    predict_gray_dir = os.path.join(save_dir, 'gray')

    if not os.path.exists(predict_gray_dir):
        print(f"!!! Predictions not found, run prediction first: {predict_gray_dir}")
        return

    results = []
    pred_files = [f for f in os.listdir(predict_gray_dir) if f.endswith('.png')]

    print(f"\n>>> Evaluation | samples: {len(pred_files)}")

    for f in pred_files:
        orig_base_name = f.replace('_gray.png', '')
        label_path = os.path.join(label_dir, orig_base_name + '.png')

        if not os.path.exists(label_path):
            continue

        pred_label = np.array(Image.open(os.path.join(predict_gray_dir, f)))
        label_map = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label_map is None:
            print(f"!!! Cannot read label: {label_path}")
            continue

        mIoU, f1, kappa, acc, prec, rec = compute_metrics(pred_label, label_map, 5)

        results.append({
            'image': orig_base_name,
            'mIoU': round(mIoU, 2), 'F1': round(f1, 2), 'Acc': round(acc, 2),
            'Precision': round(prec, 2), 'Recall': round(rec, 2), 'Kappa': round(kappa, 2)
        })

    if results:
        csv_path = os.path.join(save_dir, f"{METHOD}_ACCEV.csv")
        df = pd.DataFrame(results)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        mean_row = df[numeric_cols].mean().round(2).to_dict()
        mean_row['image'] = 'Average'
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f">>> Evaluation finished, CSV saved: {csv_path}")
        return df


def build_model(METHOD):
    if METHOD == 'UNViT':
        return UPerNetViTAdapter(backbone=ViTAdapter(interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]]),
                                 num_classes=5, backbone_indices=(0, 1, 2, 3))
    if METHOD == 'DMNet':
        return DMNet(backbone=ResNet50_vd(), num_classes=5)
    if METHOD == 'CCNet':
        return CCNet(backbone=ResNet50_vd(), num_classes=5)
    raise ValueError(f"Unsupported model: {METHOD}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['CCNet', 'DMNet', 'UNViT'])
    parser.add_argument('--weights', required=True, help='Path to the .pdparams checkpoint')
    parser.add_argument('--image_dir', required=True, help='Image file or folder')
    parser.add_argument('--output_dir', required=True, help='Root folder for results')
    parser.add_argument('--label_dir', default=None, help='Optional folder of reference labels')
    args = parser.parse_args()

    METHOD = args.model
    model = build_model(METHOD)
    OUTPUT_DIR = os.path.join(os.path.normpath(args.output_dir), METHOD)

    run_prediction(model, METHOD, args.image_dir, output_dir=OUTPUT_DIR, model_path=args.weights)
    if args.label_dir:
        run_evaluation(METHOD, args.label_dir, output_dir=OUTPUT_DIR)
