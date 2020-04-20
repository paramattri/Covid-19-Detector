import pandas as pd
import os
import shutil

original=r"D:\Work\Projects\Covid-19_model\covid-chestxray-dataset-master\images" # Change this according to your directory (Directory from which the data will be taken)
target=r"D:\Work\Projects\Covid-19_model\dataset\covid" # Directory where images will be copied from the above directory

df=pd.read_csv("covid-chestxray-dataset-master/metadata.csv")

# Taking covid-19 X-Ray images with PA view.
filter_df=df[(df['finding'] == "COVID-19") & (df['view'] == 'PA') & (df['modality'] == "X-ray")]

# Copying data from original directory to target directory
for image in filter_df['filename']:
    original_image=os.path.join(original,image)
    shutil.copy(original_image,target)
