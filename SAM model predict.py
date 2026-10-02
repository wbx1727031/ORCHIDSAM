import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from matplotlib.colors import ListedColormap


def visualize_inference(image, pred_mask, overlay_path, gray_path):
    """
    Saves the gray mask and a high-quality visual overlay.
    """
    # Save Gray Mask (1-channel label map)
    cv2.imwrite(gray_path, pred_mask.astype(np.uint8))

    # Create Visual Overlay
    # Classes: 0:None, 1:Sepal(Red), 2:Petal(Cyan), 3:Lip(Yellow), 4:Column(Green)
    custom_colors = ['none', '#FF0000', '#00FFFF', '#FFFF00', '#00FF00']
    cmap = ListedColormap(custom_colors)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(pred_mask, cmap=cmap, alpha=0.6, interpolation='nearest', vmin=0, vmax=4)
    plt.axis('off')
    plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def main():
    print("=" * 50)
    print("SAM Orchid Segmentation - Inference Tool (No GT Required)")
    print("=" * 50)

    # User Inputs
    model_path = input("1. Enter SAM checkpoint path (.pth): ").strip().strip('"')
    test_img_dir = input("2. Enter Test Images Directory: ").strip().strip('"')
    output_base = input("3. Enter Output Directory: ").strip().strip('"')

    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")

    # Load Model (Optimized for ViT-H based SAM-LOLO)
    model_type = "vit_h"
    try:
        sam_model = sam_model_registry[model_type](checkpoint=model_path, num_classes=5).to(device)
        sam_model.eval()
        predictor = SamPredictor(sam_model)
        resizer = ResizeLongestSide(sam_model.image_encoder.img_size)
        print("[*] Model loaded successfully.")
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return

    # Prepare Directories
    overlay_dir = os.path.join(output_base, "overlays")
    mask_dir = os.path.join(output_base, "masks")
    Path(overlay_dir).mkdir(parents=True, exist_ok=True)
    Path(mask_dir).mkdir(parents=True, exist_ok=True)

    img_list = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[*] Found {len(img_list)} images. Starting inference...")

    for filename in img_list:
        try:
            img_path = os.path.join(test_img_dir, filename)

            # Load and Preprocess
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                # Set image and get automated logits (Prompt-free)
                predictor.set_image(image_rgb / 255.0)
                _, _, logits = predictor.predict(return_logits=True, multimask_output=False)

                # Pixel-wise classification via Softmax & Argmax
                probs = F.softmax(logits, dim=0)
                pred_label_small = torch.argmax(probs, dim=0, keepdim=True)

                # Post-process back to original image dimensions
                input_size = resizer.apply_image(image_rgb).shape[:2]
                original_size = image.shape[:2]
                upscaled_mask = sam_model.postprocess_masks(
                    pred_label_small.float(), input_size, original_size
                ).round().squeeze().cpu().numpy()

            # Save Results
            ov_save = os.path.join(overlay_dir, filename)
            mk_save = os.path.join(mask_dir, Path(filename).stem + "_mask.png")

            visualize_inference(image_rgb, upscaled_mask, ov_save, mk_save)
            print(f"    - Processed: {filename}")

        except Exception as e:
            print(f"    [!] Skipping {filename} due to error: {e}")

    print(f"\n[DONE] Results saved in: {output_base}")


if __name__ == "__main__":
    main()