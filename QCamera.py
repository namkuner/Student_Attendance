# Python program to explain cv2.imshow() method
import shutil
# importing cv2
import cv2
import  numpy as np
# path
path = 'D21DCQCN01-Khoi.jpg'
from pathlib import Path
print(Path(path))
# Reading an image in default mode
img = cv2.imdecode(np.fromfile('WIN_20230608_16_46_40_Pro.jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
print(img.shape)
m = cv2.imread("WIN_20230608_16_46_40_Pro.jpg")
print(sum(img - m))
# Window name in which image is displayed
window_name = 'image'

# Using cv2.imshow() method
# Displaying the image
# cv2.imshow(window_name, img)
#
# # waits for user to press any key
# # (this is necessary to avoid Python kernel form crashing)
# cv2.waitKey(0)
#
# # closing all open windows
# cv2.destroyAllWindows()