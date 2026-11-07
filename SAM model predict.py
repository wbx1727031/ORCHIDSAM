import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 5
COLORMAP = np.array(
    [[0, 0, 0], [255, 0, 0], [0, 255, 255], [255, 255, 0], [0, 255, 0]],
    dtype=np.uint8,
)


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


def visualize_prediction(image, label, mask, save_path, gray_save_path):
    """Save the gray prediction and an overlay; return the label map converted from the RGB label (or None)."""
    if len(image.shape) not in (2, 3):
        raise ValueError(f"Invalid image shape: {image.shape}")
    if mask.ndim != 2:
        raise ValueError(f"Invalid mask shape: {mask.shape}")

    label_map = None
    if label is not None:
        if label.shape[-1] != 3:
            raise ValueError(f"Invalid label shape: {label.shape}")
        color_to_label = {tuple(int(v) for v in COLORMAP[i]): i for i in range(1, len(COLORMAP))}
        label_map = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
        for color, label_id in color_to_label.items():
            label_map[np.all(label == color, axis=-1)] = label_id

    cv2.imwrite(gray_save_path, mask.astype(np.uint8))
    try:
        custom_colors = ['none', '#FF0000', '#00FFFF', '#FFFF00', '#00FF00']
        new_cmap = ListedColormap(custom_colors)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(mask, cmap=new_cmap, alpha=0.65, interpolation='nearest', vmin=0, vmax=4)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300,
                    facecolor='white', edgecolor='none')
        plt.close()
    except Exception as e:
        raise RuntimeError(f"Failed to save image: {e}")
    return label_map


def build_model(sam_checkpoint, model_type='vit_h', num_classes=NUM_CLASSES):
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint, num_classes=num_classes).to(device=device)
    sam_model.eval()
    return sam_model


def predict_single_image(sam_model, sam_trans, image_rgb):
    original_size = image_rgb.shape[:2]
    with torch.no_grad():
        resize_image = sam_trans.apply_image(image_rgb).transpose((2, 0, 1))
        image_tensor = torch.tensor(resize_image / 255, dtype=torch.float32).unsqueeze(0)
        image_embedding = sam_model.image_encoder(image_tensor.to(device))
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        input_size = tuple(resize_image.shape[1:])
        upscaled_logits = sam_model.postprocess_masks(
            low_res_masks, input_size, original_size
        ).to(device)
        probs = F.softmax(upscaled_logits.squeeze(), dim=0)
        pred_label = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)
    return pred_label


def main():
    parser = argparse.ArgumentParser(description="Prompt-free floral organ segmentation with adapted SAM")
    parser.add_argument('--mode', required=True, choices=['SAM', 'SAM&LOSS', 'SAM_LOSS'],
                        help='SAM: fine-tuned without the optimized loss; SAM&LOSS: fine-tuned with the optimized loss '
                             '(quote it in the shell: "SAM&LOSS"; SAM_LOSS is accepted as an alias)')
    parser.add_argument('--weights', required=True, help='Path to the .pth checkpoint matching --mode')
    parser.add_argument('--image_dir', required=True, help='Folder of input images')
    parser.add_argument('--output_dir', required=True, help='Root folder for results')
    parser.add_argument('--label_dir', default=None,
                        help='Optional folder of RGB reference labels (same file names as the images)')
    parser.add_argument('--model_type', default='vit_h', help='SAM backbone type')
    args = parser.parse_args()

    METHOD = 'SAM&LOSS' if args.mode == 'SAM_LOSS' else args.mode
    output_base = os.path.join(os.path.normpath(args.output_dir), METHOD)
    overlay_output_dir = os.path.join(output_base, "overlay")
    gray_output_dir = os.path.join(output_base, "grey")

    print(f"Using device: {device}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weight file not found: {args.weights}")

    sam_model = build_model(args.weights, model_type=args.model_type, num_classes=NUM_CLASSES)
    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)

    Path(overlay_output_dir).mkdir(parents=True, exist_ok=True)
    Path(gray_output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for filename in sorted(os.listdir(args.image_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(args.image_dir, filename)
        print(f"Processing {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            print(f"!!! Cannot read image, skipped: {image_path}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_rgb = None
        if args.label_dir:
            label_bgr = cv2.imread(os.path.join(args.label_dir, filename))
            if label_bgr is None:
                print(f"!!! Cannot read label, skipped: {filename}")
                continue
            label_rgb = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2RGB)

        pred_label = np.squeeze(predict_single_image(sam_model, sam_trans, image_rgb))
        if pred_label.ndim != 2:
            raise RuntimeError(
                f"Unexpected prediction shape {pred_label.shape}; the mask decoder must output "
                f"{NUM_CLASSES} channels when multimask_output=False."
            )

        overlay_save_path = os.path.join(overlay_output_dir, filename)
        gray_save_path = os.path.join(gray_output_dir, Path(filename).stem + "_grey.png")
        label_map = visualize_prediction(image_rgb, label_rgb, pred_label, overlay_save_path, gray_save_path)

        if label_map is not None:
            mIoU, f1, kappa, acc, prec, rec = compute_metrics(pred_label, label_map, NUM_CLASSES)
            results.append({
                'image': filename,
                'mIoU': round(mIoU, 2),
                'F1': round(f1, 2),
                'Acc': round(acc, 2),
                'Precision': round(prec, 2),
                'Recall': round(rec, 2),
                'Kappa': round(kappa, 2),
            })

    if not results:
        print(f"Predictions saved to: {output_base}")
        return

    csv_path = os.path.join(output_base, METHOD + '_ACCEV.csv')
    df = pd.DataFrame(results)
    numeric_cols = ['mIoU', 'F1', 'Acc', 'Precision', 'Recall', 'Kappa']
    mean_row = df[numeric_cols].mean().round(2).to_dict()
    mean_row['image'] = 'Average'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
