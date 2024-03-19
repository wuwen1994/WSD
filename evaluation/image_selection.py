import cv2 as cv
import os


target_dir = "C:/Users/Wu/Desktop/weakly/results/SBU/"
save_dir = "big_error_image_test"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

all_image_list = []
with open("b.txt", 'r') as f:
    content = f.readlines()
    for c in content:
        all_image_list.append(c.strip())
for image_name in all_image_list:
    image = cv.imread(target_dir+image_name)
    image_name = image_name.replace(".png", "_p.png")
    cv.imwrite(os.path.join(save_dir, image_name), image)
