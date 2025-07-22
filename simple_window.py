import cv2
import os

# 1. Point to a real image file—absolute or relative path
img_path = "bus.jpg"  # ← make sure this file actually exists here
if not os.path.isfile(img_path):
    print(f"❌ File not found: {img_path}")
    exit(1)

img = cv2.imread(img_path)
if img is None:
    print(f"❌ Failed to read image: {img_path}")
    exit(1)

# 2. Show it
cv2.imshow("test", img)
print("Showing image, press any key in the window to exit")
cv2.waitKey(0)
cv2.destroyAllWindows()
