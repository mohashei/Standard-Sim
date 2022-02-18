"""
Gets mean and std dev of pixel values in rgb images to check for outliers.
"""
import os
import numpy as np
from PIL import Image
from glob import glob

def find_files(root):
    imgs = sorted([filename for filename in glob(f"{root}/*.png") if "change-0" in filename])
    files = []
    for name in imgs:
        label_file1 = name.replace(".png", "-segmentation0001.exr")
        label_file2 = label_file1.replace("change-0", "change-1")
        label_json1 = name.replace(".png", "_label.json")
        label_json2 = label_json1.replace("change-0", "change-1")
        files_exist = (
            os.path.isfile(label_file1)
            and os.path.isfile(label_file2)
            and os.path.isfile(label_json1)
            and os.path.isfile(label_json2)
        )

        if files_exist:
            bbox_json = name.replace("_change-0.png", "-boxes.json")
            label_file = name.replace("_change-0.png", "-label.png")
            files.append(
                {
                    "label1": label_file1,
                    "label2": label_file2,
                    "label": label_file,
                    "label1_json": label_json1,
                    "label2_json": label_json2,
                    "bbox_json": bbox_json,
                }
            )
    return files

if __name__ == "__main__":
	data_root = "/app/renders_multicam_diff_1"


	files = find_files(data_root)

	mean_diffs = []
	for f in files:
		mean1 = np.mean(np.array(Image.open(f['label'].replace('-label.png', '_change-0.png'))))
		mean2 = np.mean(np.array(Image.open(f['label'].replace('-label.png', '_change-1.png'))))
		mean_diffs.append(mean2-mean1)

	overall_mean_pixel_diff = np.mean(mean_diffs)
	overall_std_pixel_diff = np.std(mean_diffs)
	print("overall mean pixel diff ", overall_mean_pixel_diff, " overall std pixel diff ", overall_std_pixel_diff)
	
	fns = []
	for f in files:
		mean1 = np.mean(np.array(Image.open(f['label'].replace('-label.png', '_change-0.png'))))
		mean2 = np.mean(np.array(Image.open(f['label'].replace('-label.png', '_change-1.png'))))
		if mean2 - mean1 > overall_mean_pixel_diff+overall_std_pixel_diff:
			fns.append(f['label'])
		elif mean2 - mean1 < overall_mean_pixel_diff-overall_std_pixel_diff:
			fns.append(f['label'])

	print(fns)
	print(len(fns))
