import os
import shutil
import random

original=r"D:\Work\Projects\Covid-19_model\chest_xray\chest_xray\train\NORMAL" # Change this according to your directory (Directory from which the data will be taken)
target=r"D:\Work\Projects\Covid-19_model\dataset\normal" # Directory where images will be copied from the above directory

file_path=[]

for r,d,f in os.walk(original):
    for file in f:
        if '.jpeg' in file:
            file_path.append(os.path.join(r,file))

random.seed(42)
random.shuffle(file_path)

# Taking 99 images only as we have only 99 images of Covid-19 X-ray
# So, class imbalance will not take place.
image_path=file_path[0:99]

# Copying the images to target destination.
for image in image_path:
    shutil.copy(image,target)
