#!/usr/bin/env python3

import argparse
import os
import cv2
import numpy as np

def negative_transform(img):
    """Apply negative transformation."""
    return 255 - img

def gamma_transform(img, gamma=0.5):
    """Power-law (gamma) transformation."""
    img_float = img / 255.0
    img_gamma = np.power(img_float, gamma)
    return np.uint8(img_gamma * 255)

def main():
    parser = argparse.ArgumentParser(description="Image Enhancement Script")
    parser.add_argument("input", help="Path to input image")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {args.input}")

    print("Choose enhancement to apply:")
    print("1. Negative")
    print("2. Gamma")
    print("3. Both Negative + Gamma")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        enhanced = negative_transform(img)
    elif choice == "2":
        gamma_val = float(input("Enter gamma value (e.g., 0.5, 1.5): "))
        enhanced = gamma_transform(img, gamma=gamma_val)
    elif choice == "3":
        gamma_val = float(input("Enter gamma value for Gamma transform: "))
        enhanced = gamma_transform(img, gamma=gamma_val)
        enhanced = negative_transform(enhanced)
    else:
        print("Invalid choice. Exiting.")
        return

    in_dir = os.path.dirname(os.path.abspath(args.input))
    out_dir = os.path.join(in_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    filename = os.path.basename(args.input)
    name, ext = os.path.splitext(filename)
    out_path = os.path.join(out_dir, f"enh_{name}{ext}")

    cv2.imwrite(out_path, enhanced)
    print(f"Enhanced image saved to: {out_path}")

if __name__ == "__main__":
    main()
