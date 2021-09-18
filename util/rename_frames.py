import os, os.path as op
import shutil

PATH = 'D:\\edu\\UniBonn\\Study\\thesis\\codes\\results\\slides\\lego_title\\3'
TARGET_PATH = op.abspath(op.join(PATH, '..\\res'))

frames = [f for f in os.listdir(PATH) if op.isfile(op.join(PATH, f))]

os.makedirs(TARGET_PATH, exist_ok=True)
for frame in frames:
	filename, ext = op.splitext(frame)
	shutil.copy2(op.join(PATH, frame), op.join(TARGET_PATH, "{:04d}{}".format(int(filename)+72*2, ext)))