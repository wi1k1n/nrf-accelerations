import os, os.path as op, re, json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2 as cv

PATH_RAW_DATA = '../realdata/flower_dome/'
VOXEL_NUMS = 64
PATH_OUT = op.abspath(op.join(PATH_RAW_DATA, 'dataset'))

fnameMeas = op.abspath(PATH_RAW_DATA + '/meas.xml')
assert op.isfile(fnameMeas), 'File meas.xml is not found!'

pathMeasFolder = op.abspath(PATH_RAW_DATA + '/meas/')
pathMeasFolder = [op.join(pathMeasFolder, name) for name in os.listdir(pathMeasFolder) if op.isdir(op.join(pathMeasFolder, name))]
assert len(pathMeasFolder), 'Folder with measurements is not found'
pathMeasFolder = pathMeasFolder[0]

os.makedirs(PATH_OUT, exist_ok=True)

print('Processing measurements at paths:')
print(fnameMeas)
print(pathMeasFolder)
print('Out folder:', PATH_OUT)

# Parse meas.xml file and get zoom levels
tree = ET.parse(op.abspath(fnameMeas))
root = tree.getroot()
camConfigs = root.find('cameras')

# dict of zoom levels
# --- each zoom level is a dict of cameras
zoomLevels = dict()
for camConf in camConfigs:
	idx, phi, theta = [camConf.attrib[k] for k in camConf.attrib.keys()]
	calib = camConf.find('calib3d')
	for zoom in calib:
		zmLevel = zoom.attrib.get('zoom')
		calibData = [l.strip() for l in zoom.text.split('OpenCV ')[1].splitlines() if l.strip()]
		intrinsic = np.fromstring(' '.join(calibData[:3]), sep=' ').reshape((3, 3))
		distort = np.fromstring(calibData[3], sep=' ')
		translation = np.fromstring(calibData[4], sep=' ')
		rotation = np.fromstring(calibData[5], sep=' ')

		if not zmLevel in zoomLevels:
			zoomLevels[zmLevel] = dict()
		if not idx in zoomLevels[zmLevel]:
			zoomLevels[zmLevel][idx] = dict()
		zoomLevels[zmLevel][idx] = {
			'cam_idx': idx,
			'phi': phi,
			'theta': theta,
			'intrinsic': intrinsic,
			'distort': distort,
			'translation': translation,
			'rotation': rotation
		}


print('Zoom levels: ', list(zoomLevels.keys()))
zoomLevel = list(zoomLevels.keys())[0]
print('Cameras #{}: [{}] ... [{}]'.format(len(zoomLevels[zoomLevel].keys()), list(zoomLevels[zoomLevel].keys())[0], list(zoomLevels[zoomLevel].keys())[-1]))


##### Create dataset
# Iterate zoom levels
measurementFNames = os.listdir(pathMeasFolder)
for zoomIdx, cameras in zoomLevels.items():
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
	print('>> Processing zoom level: ', zoomIdx)
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	curFolder = op.abspath(op.join(PATH_OUT, 'zoom_' + str(zoomIdx)))
	os.makedirs(curFolder, exist_ok=True)
	print('Outdir: ', curFolder)

	pathRGB = op.join(curFolder, 'rgb')
	pathPose = op.join(curFolder, 'pose')
	pathPosePL = op.join(curFolder, 'pose_pl')
	os.makedirs(pathRGB, exist_ok=True)
	os.makedirs(pathPose, exist_ok=True)
	os.makedirs(pathPosePL, exist_ok=True)

	json_data = {'frames': []}
	camPoints = []
	for camIdx, cam in cameras.items():
		# Searching for measurement file in folder
		print('Cam #{}/{}'.format(camIdx, len(cameras.keys())))
		lightIdx = 133
		lightPhi = 270
		lightTheta = 75
		regText = '^cv0{0,2}'+str(camIdx)+'_tv0{0,2}'+str(cam['theta'])+'(\.)?(\d{0,2})?_pv0{0,2}'+str(cam['phi'])+'(\.)?(\d{0,2})?_cl0{0,2}'+str(lightIdx)+'_tl0{0,2}'+str(lightTheta)+'(\.)?(\d{0,2})?_pl0{0,2}'+str(lightPhi)+'(\.)?(\d{0,2})?_ISO400_FQ0_IDX1\.jpg$'
		regex = re.compile(regText)
		measFile = [fn for i, fn in enumerate(measurementFNames) if regex.match(fn)]
		# assert len(measFile) == 1, 'Either measurement file is not found or found more than one corresponding files. Regex: ' + regText
		if len(measFile) != 1:
			print('Measurement cv{}_tv{}_pv{}_cl{}_tl{}_pl{} not found!'.format(camIdx, cam['theta'], cam['phi'], lightIdx, lightTheta, lightPhi))
			continue
		measPath = op.abspath(op.join(pathMeasFolder, measFile[0]))


		# Prepare extrinsics retrieved from meas.xml file
		extrinsics = np.concatenate((
			np.concatenate((np.array(Rot.from_rotvec(cam['rotation']).as_matrix()),
							cam['translation'][:, None],), axis=1),
			np.r_[0, 0, 0, 1][None]), axis=0)
		extrinsics[2, :] *= -1.  # invert camera directions
		camPoints.append((np.linalg.inv(extrinsics) @ np.r_[0, 0, 0, 1])[:3])


		# Add data for transforms.json file (used for light/cam visualization)
		frame_data = {
			'file_path': measPath,
			'transform_matrix': np.linalg.inv(extrinsics).tolist(),
			'pl_transform_matrix': np.concatenate((np.zeros((3, 4)), np.r_[0, 0, 0, 1][None]), axis=0).tolist()
		}
		json_data['frames'].append(frame_data)



		# Load image, undistort, (resize?) and save to dataset
		# img = cv.imread(measPath)
		# undistortedImg = cv.undistort(img, cam['intrinsic'], cam['distort'])
		# cv.imwrite(op.join(pathRGB, '{:04d}.jpg'.format(int(camIdx))), undistortedImg)


		with open(op.join(pathPose, '{:04d}.txt'.format(int(camIdx))), 'w') as fo:
			for ii, pose in enumerate(frame_data['transform_matrix']):
				print(" ".join([str(-p) if (((j == 2) | (j == 1)) and (ii < 3)) else str(p)
								for j, p in enumerate(pose)]), file=fo)

		with open(op.join(pathPosePL, '{:04d}.txt'.format(int(camIdx))), 'w') as fo:
			for ii, pose in enumerate(frame_data['pl_transform_matrix']):
				print(" ".join([str(-p) if (((j == 2) | (j == 1)) and (ii < 3)) else str(p)
								for j, p in enumerate(pose)]), file=fo)

	# Writing intrinsics from the last camera (since it is the same in all cameras)
	np.savetxt(op.join(curFolder, 'intrinsics.txt'), cam['intrinsic'])

	# Estimate bbox simply by taking min/max coordinates of camera positions
	camPoints = np.array(camPoints)
	bbox = camPoints.min(axis=0).tolist() + camPoints.max(axis=0).tolist()
	if bbox[5] < 0: bbox[5] = 0  # extend bbox to 0
	bbox[2] = -30
	voxel_size = ((bbox[3]-bbox[0]) * (bbox[4]-bbox[1]) * (bbox[5]-bbox[2]) / VOXEL_NUMS) ** (1/3)
	with open(op.join(curFolder, 'bbox.txt'), 'w') as out_file:
		print(" ".join(['{:.5f}'.format(f) for f in bbox + [voxel_size]]), file=out_file)

	with open(op.join(curFolder, 'transforms.json'), 'w') as out_file:
		json.dump(json_data, out_file, indent=4)