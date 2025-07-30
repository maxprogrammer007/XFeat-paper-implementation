import torch
import cv2
import numpy as np
import os
import sys

# Import the XFeat model from the modules folder
from modules.xfeat import XFeat

# Helper function to plot matches
def plot_matches(img0, img1, kps0, kps1, matches, color=(0,255,0), radius=3):
    if isinstance(img0, str):
        img0 = cv2.imread(img0)
        img1 = cv2.imread(img1)
    if img0.ndim == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    
    canvas = np.zeros((max(h0,h1), w0+w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:] = img1

    if kps0 is not None:
        for i in range(len(kps0)):
            x,y = kps0[i].astype(int)
            cv2.circle(canvas, (x,y), radius, color, -1)
    if kps1 is not None:
        for i in range(len(kps1)):
            x,y = kps1[i].astype(int)
            cv2.circle(canvas, (x+w0,y), radius, color, -1)
    if matches is not None:
        for i in range(len(matches)):
            idx0, idx1 = matches[i]
            pt0 = kps0[idx0].astype(int)
            pt1 = kps1[idx1].astype(int)
            cv2.line(canvas, tuple(pt0), (pt1[0]+w0, pt1[1]), color, 1)

    return canvas

# --- Main part of the script ---
if __name__ == '__main__':
    # --- 1. Setup ---
    # The XFeat class now handles loading the weights in its __init__
    # We can use the default weights by just creating an instance.
    model = XFeat()
    
    # --- 2. Load Images ---
    img1_path = os.path.join('assets', 'ref.png')
    img2_path = os.path.join('assets', 'tgt.png')
    
    img1_bgr = cv2.imread(img1_path)
    img2_bgr = cv2.imread(img2_path)

    if img1_bgr is None or img2_bgr is None:
        print("Error: Could not load images. Please check that the file paths are correct")
        print(f"Attempted to load: {os.path.abspath(img1_path)} and {os.path.abspath(img2_path)}")
        sys.exit()

    # --- 3. Match features using the correct function ---
    print("Matching features...")
    
    # **FIX:** Call the correct 'match_xfeat' function
    mkpts0, mkpts1 = model.match_xfeat(img1_bgr, img2_bgr)
    
    print(f"Found {len(mkpts0)} matches.")
    
    # --- 4. Visualize the results ---
    # The function returns matched keypoints directly. We need to create
    # a simple list of indices for the visualization function.
    matches_indices = np.arange(len(mkpts0)).reshape(-1,1)
    matches_indices = np.hstack((matches_indices, matches_indices))

    output_image = plot_matches(img1_bgr, img2_bgr, mkpts0, mkpts1, matches_indices)
    
    cv2.imwrite('matches_output.png', output_image)
    print("Saved visualization to 'matches_output.png'")

    cv2.imshow('XFeat Matches', output_image)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()