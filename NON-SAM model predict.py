import os
import shutil
import paddle
import numpy as np
from PIL import Image
from paddleseg.models import ResNet50_vd, CCNet, DMNet, UPerNetViTAdapter
from pathlib import Path


# ========== Image List Retrieval ==========
def get_image_list(image_path):
    valid_suffix = ['.jpeg', '.jpg', '.bmp', '.png']
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
    else:
        raise FileNotFoundError(f"Path does not exist: {image_path}")

    if not image_list:
        raise RuntimeError("No valid image files found in the path.")
    return image_list


# ========== Core Prediction without Evaluation ==========
def run_inference(model, model_name, image_list, output_root, weight_path):
    save_dir = os.path.join(output_root, "Inference_Results", model_name)

    # Refresh output directory
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    Path(os.path.join(save_dir, 'masks')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(save_dir, 'overlays')).mkdir(parents=True, exist_ok=True)

    # Load Model Weights
    try:
        state_dict = paddle.load(weight_path)
        model.set_dict(state_dict)
        print(f"Weights loaded successfully from: {weight_path}")
    except Exception as e:
        raise RuntimeError(f"Weight loading failed: {e}")

    model.eval()

    # Colormap: Background(0), Sepal(1), Petal(2), Lip(3), Column(4)
    colormap = np.array([
        [0, 0, 0],  # Background
        [255, 0, 0],  # Red (Sepal)
        [0, 255, 255],  # Cyan (Petal)
        [255, 255, 0],  # Yellow (Lip)
        [0, 255, 0]  # Green (Column)
    ], dtype=np.uint8)

    print(f"Starting inference using {model_name}...")

    for img_path in image_list:
        img_name = os.path.basename(img_path)
        try:
            # Preprocessing
            img_pil = Image.open(img_path).convert('RGB')
            img_raw = np.array(img_pil)
            img_tensor = paddle.to_tensor(img_raw.astype('float32') / 255.0).transpose((2, 0, 1)).unsqueeze(0)

            # Prediction
            with paddle.no_grad():
                pred = model(img_tensor)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                pred_label = paddle.argmax(pred, axis=1).squeeze().numpy().astype(np.uint8)

            # Save Gray Mask
            mask_path = os.path.join(save_dir, 'masks', os.path.splitext(img_name)[0] + '_mask.png')
            Image.fromarray(pred_label).save(mask_path)

            # Save Visual Overlay
            overlay = (img_raw * 0.5 + colormap[pred_label] * 0.5).astype(np.uint8)
            overlay_path = os.path.join(save_dir, 'overlays', os.path.splitext(img_name)[0] + '_overlay.png')
            Image.fromarray(overlay).save(overlay_path)

            print(f"Processed: {img_name}")

        except Exception as e:
            print(f"Skipping {img_name} due to error: {e}")

    print(f"\n[Task Completed] Results are stored in: {save_dir}")


# ========== Entry Point ==========
if __name__ == '__main__':
    print("===== Advanced Orchid Segmentation Inference System =====")

    # User Inputs
    img_dir = input("Enter Input Images Directory: ").strip().strip('"')
    weight_path = input("Enter Model Weights Path (.pdparams): ").strip().strip('"')
    out_root = input("Enter Output Directory: ").strip().strip('"')

    print("\nSelect Architecture:")
    print("1. CCNet (Criss-Cross Attention Network)")
    print("2. DMNet (Dynamic Multiscale Network)")
    print("3. UNViT (UPerNet with ViT Adapter) [Default]")
    choice = input("Choice (1-3): ")

    # Model Selection
    if choice == '1':
        model = CCNet(backbone=ResNet50_vd(), num_classes=5)
        method = 'CCNet'
    elif choice == '2':
        model = DMNet(backbone=ResNet50_vd(), num_classes=5)
        method = 'DMNet'
    else:
        # Default to UPerNetViTAdapter
        model = UPerNetViTAdapter(backbone=ResNet50_vd(), num_classes=5, backbone_indices=(0, 1, 2))
        method = 'UNViT'

    # Run Process
    try:
        img_list = get_image_list(img_dir)
        run_inference(model, method, img_list, out_root, weight_path)
    except Exception as e:
        print(f"\n[Error]: {e}")