import os, os.path as op, re, json, random
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2 as cv

VOXEL_NUMS = 64
INTREXTR_FROM_MEASXML = True

XML_MEAS_PATH = '../realdata/flower_dome/meas.xml'
# RAW_DATA_FOLDER = '../realdata/flower_dome/meas/tv000_045_cl133'
# OUTPUT_FOLDER = '../realdata/flower_dome/dataset/'

EXTR_MEAS_PATH = '../realdata/flower_dome/meas/tv000_045_cl133_masked/extrinsics.json'
INTR_MEAS_PATH = '../realdata/flower_dome/meas/tv000_045_cl133_masked/intrinsics.json'
RAW_DATA_FOLDER = '../realdata/flower_dome/meas/tv000_045_cl133_masked'
OUTPUT_FOLDER = '../realdata/flower_dome/dataset_png/'

NAME_POSTFIX = '_masked'
EXTENSION = 'png'  # 'jpg'
PROCESS_IMAGES = False
SHUFFLE = True
# BBOX = [-0.75, -0.75, -1.5, 0.75, 0.75, -0.5]
# BBOX = [-50, -50, -10, 50, 50, 80]
BBOX = [-30, -30, -10, 30, 30, 70]
# ROTATE = [0, -1, -1]
UNDO_CV1 = True
INVERT_TRANSLATION = False

random.seed(1)
np.random.seed(1)
print('Processing measurements at path:')
print(RAW_DATA_FOLDER)
print('Out folder:', OUTPUT_FOLDER)


RAW_DATA_FOLDER = op.abspath(RAW_DATA_FOLDER)
OUTPUT_FOLDER = op.abspath(OUTPUT_FOLDER)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
assert op.isdir(RAW_DATA_FOLDER), 'Folder with measurements is not found'

# dict of zoom levels
# --- each zoom level is a dict of cameras
zoomLevels = dict()

##### Read extrinsics/intrinsics from meta files
# If getting intrinsics/extrinsics from meas.xml
if INTREXTR_FROM_MEASXML:
	XML_MEAS_PATH = op.abspath(XML_MEAS_PATH)
	assert op.isfile(XML_MEAS_PATH), 'File meas.xml is not found!'

	print('Processing XML measurement file:')
	print(XML_MEAS_PATH)

	# Parse meas.xml file and get zoom levels
	tree = ET.parse(op.abspath(XML_MEAS_PATH))
	root = tree.getroot()
	camConfigs = root.find('cameras')

	# iterate over cameras
	for camConf in camConfigs:
		idx, phi, theta = [camConf.attrib[k] for k in camConf.attrib.keys()]
		calib = camConf.find('calib3d')
		for zoom in calib:
			zmLevel = zoom.attrib.get('zoom')
			calibData = [l.strip() for l in zoom.text.split('OpenCV ')[1].splitlines() if l.strip()]
			intrinsic = np.fromstring(' '.join(calibData[:3]), sep=' ').reshape((3, 3))
			distort = np.fromstring(calibData[3], sep=' ')
			translation = -np.fromstring(calibData[4], sep=' ')
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
# if getting extrinsics/intrinsics from *.json files (meshroom + raytracing)
else:
	EXTR_MEAS_PATH = op.abspath(EXTR_MEAS_PATH)
	INTR_MEAS_PATH = op.abspath(INTR_MEAS_PATH)
	assert op.isfile(EXTR_MEAS_PATH) and op.isfile(INTR_MEAS_PATH), '*.json files are not found!'

	zoomLevels['-1'] = dict()

	# name_postfix = '_masked'

	# get intrinsics
	with open(INTR_MEAS_PATH) as fh:
		intrinsics = np.array(json.load(fh))

	# get extrinsics
	with open(EXTR_MEAS_PATH) as fh:
		extr = json.load(fh)
	for k, v in extr.items():
		idx = int(k[2:])
		if not k in zoomLevels['-1']:
			zoomLevels['-1'][str(idx)] = dict()
		zoomLevels['-1'][str(idx)] = {
			'cam_idx': idx,
			'phi': '\d{3}',
			'theta': '\d{3}',
			'intrinsic': intrinsics,
			'distort': None,
			'translation': None,
			'rotation': None,
			'extrinsics': np.array(v)
		}

	# still need distort vector
	tree = ET.parse(op.abspath(XML_MEAS_PATH))
	root = tree.getroot()
	camConfigs = root.find('cameras')

	# iterate over cameras
	for camConf in camConfigs:
		idx, phi, theta = [camConf.attrib[k] for k in camConf.attrib.keys()]
		idx = str(int(idx))
		if not idx in zoomLevels['-1']:
			continue
		calib = camConf.find('calib3d')
		for zoom in calib:
			zmLevel = zoom.attrib.get('zoom')
			calibData = [l.strip() for l in zoom.text.split('OpenCV ')[1].splitlines() if l.strip()]
			distort = np.fromstring(calibData[3], sep=' ')
			zoomLevels['-1'][idx].update({
				'distort': distort
			})
			break

print('Zoom levels: ', list(zoomLevels.keys()))
zoomLevel = list(zoomLevels.keys())[0]
print('Cameras #{}: [{}] ... [{}]'.format(len(zoomLevels[zoomLevel].keys()), list(zoomLevels[zoomLevel].keys())[0], list(zoomLevels[zoomLevel].keys())[-1]))






##### Create dataset
# Iterate zoom levels
measurementFNames = os.listdir(RAW_DATA_FOLDER)
for zoomIdx, cameras in zoomLevels.items():
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
	print('>> Processing zoom level: ', zoomIdx)
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	curFolder = op.abspath(op.join(OUTPUT_FOLDER, 'zoom_' + str(zoomIdx)))
	os.makedirs(curFolder, exist_ok=True)
	print('Outdir: ', curFolder)

	pathRGB = op.join(curFolder, 'rgb')
	pathPose = op.join(curFolder, 'pose')
	pathPosePL = op.join(curFolder, 'pose_pl')
	os.makedirs(pathRGB, exist_ok=True)
	os.makedirs(pathPose, exist_ok=True)
	os.makedirs(pathPosePL, exist_ok=True)

	adjust_ext = np.eye(4)
	if 'UNDO_CV1' in locals() and UNDO_CV1:
		cam1 = cameras['1']
		if 'extrinsics' in cam1 and cam1['extrinsics'] is not None:
			ext1 = cam1['extrinsics']
		else:
			ext1 = np.concatenate((
				np.concatenate((np.array(Rot.from_rotvec(cam1['rotation']).as_matrix()),
								cam1['translation'][:, None],), axis=1),
				np.r_[0, 0, 0, 1][None]), axis=0)
		adjust_ext = ext1

	json_data = {'frames': []}
	camPoints = []
	measImgPaths = {}  # paths to real existing images

	# Create list of available images
	for camIdx, cam in cameras.items():
		# Searching for measurement file in folder
		lightIdx = 133
		lightPhi = 270
		lightTheta = 75
		regText = '^cv0{0,2}' + str(camIdx) +'_tv0{0,2}' + str(cam['theta']) +'(\.)?(\d{0,2})?_pv0{0,2}' + str(cam['phi']) +'(\.)?(\d{0,2})?_cl0{0,2}' + str(lightIdx) +'_tl0{0,2}' + str(lightTheta) +'(\.)?(\d{0,2})?_pl0{0,2}' + str(lightPhi) +'(\.)?(\d{0,2})?_ISO400_FQ0_IDX1' + NAME_POSTFIX + '\.' + EXTENSION + '$'
		regex = re.compile(regText)
		measFile = [fn for i, fn in enumerate(measurementFNames) if regex.match(fn)]
		# assert len(measFile) == 1, 'Either measurement file is not found or found more than one corresponding files. Regex: ' + regText
		if len(measFile) != 1:
			print('Measurement cv{}_tv{}_pv{}_cl{}_tl{}_pl{} not found!'.format(camIdx, cam['theta'], cam['phi'], lightIdx, lightTheta, lightPhi))
			continue
		measPath = op.abspath(op.join(RAW_DATA_FOLDER, measFile[0]))
		measImgPaths[camIdx] = measPath

	saveIdcs = [i for i in range(len(measImgPaths.keys()))]

	if 'SHUFFLE' in locals() and SHUFFLE:
		random.shuffle(saveIdcs)

	# Loop for processing images
	for camIdx, cam in cameras.items():
		print('Cam #{}/{}{}{}'.format(camIdx,
									  len(cameras.keys()),
									  ' --> {}'.format(saveIdcs[0]) if len(saveIdcs) else '',
									  ('' if camIdx in measImgPaths else ' skipped!')))
		if not camIdx in measImgPaths:
			continue


		measPath = measImgPaths[camIdx]

		if 'extrinsics' in cam and cam['extrinsics'] is not None:
			extrinsics = cam['extrinsics']
		else:
			# Prepare extrinsics retrieved from meas.xml file
			extrinsics = np.concatenate((
				np.concatenate((np.array(Rot.from_rotvec(cam['rotation']).as_matrix()),
								cam['translation'][:, None],), axis=1),
				np.r_[0, 0, 0, 1][None]), axis=0)

		# Invert camera directions
		# rm = np.eye(3)
		# rm[2, 2] = -1

		extrinsics = adjust_ext @ extrinsics

		if 'INVERT_TRANSLATION' in locals() and INVERT_TRANSLATION:
			extrinsics[:3, 3] *= -1



		# if 'ROTATE' in locals():
		# 	rotM = np.concatenate((Rot.from_rotvec(ROTATE).as_matrix(), np.r_[0, 0, 1][None]), axis=0)
		# 	rotM = np.concatenate((rotM, np.zeros(4)[:, None]), axis=1)
		# 	extrinsics = rotM @ extrinsics


		camPoint = (extrinsics @ np.r_[0, 0, 0, 1])[:3]
		# if 'ROTATE' in locals():
		# 	camPoint = Rot.from_rotvec(ROTATE).as_matrix() @ camPoint
		camPoints.append(camPoint)


		# Add data for transforms.json file (used for light/cam visualization)
		frame_data = {
			'file_path': measPath,
			'transform_matrix': np.linalg.inv(extrinsics).tolist(),
			'pl_transform_matrix': np.concatenate((np.zeros((3, 4)), np.r_[0, 0, 0, 1][None]), axis=0).tolist()
		}
		json_data['frames'].append(frame_data)



		# Load image, undistort, (resize?) and save to dataset
		if 'PROCESS_IMAGES' in locals() and PROCESS_IMAGES:
			img = cv.imread(measPath, cv.IMREAD_UNCHANGED)
			undistortedImg = cv.undistort(img, cam['intrinsic'], cam['distort'])
			# cv.imwrite(op.join(pathRGB, '{:04d}.{}'.format(int(camIdx), EXTENSION)), undistortedImg)
			cv.imwrite(op.join(pathRGB, '{:04d}.{}'.format(saveIdcs[0], EXTENSION)), undistortedImg)

		# cv.imwrite(op.join(pathRGB, '{:04d}.jpg'.format(int(camIdx))), img)

		with open(op.join(pathPose, '{:04d}.txt'.format(saveIdcs[0])), 'w') as fo:
			for ii, pose in enumerate(frame_data['transform_matrix']):
				print(" ".join([str(-p) if (((j == 2) | (j == 1)) and (ii < 3)) else str(p)
								for j, p in enumerate(pose)]), file=fo)

		with open(op.join(pathPosePL, '{:04d}.txt'.format(saveIdcs[0])), 'w') as fo:
			for ii, pose in enumerate(frame_data['pl_transform_matrix']):
				print(" ".join([str(-p) if (((j == 2) | (j == 1)) and (ii < 3)) else str(p)
								for j, p in enumerate(pose)]), file=fo)

		saveIdcs.pop(0)

	assert len(camPoints), 'No files have been processed'

	# Writing intrinsics from the last camera (since it is the same in all cameras for one zoom level)
	np.savetxt(op.join(curFolder, 'intrinsics.txt'), cam['intrinsic'])

	if 'BBOX' in locals():
		bbox = BBOX
	else:
		# Estimate bbox simply by taking min/max coordinates of camera positions
		camPoints = np.array(camPoints)
		bbox = camPoints.min(axis=0).tolist() + camPoints.max(axis=0).tolist()
		# large bbox: 1.4/-0.5/0.66
		shrinkRate = 0.6
		bbox = [b * (1 if i % 3 == 2 else shrinkRate) for i, b in enumerate(bbox)]
		bbox[2] = -bbox[5] * 0.1
		bbox[5] = bbox[5] * 0.36
	if 'VOXEL_SIZE' in locals():
		voxel_size = VOXEL_SIZE
	else:
		voxel_size = ((bbox[3]-bbox[0]) * (bbox[4]-bbox[1]) * (bbox[5]-bbox[2]) / VOXEL_NUMS) ** (1/3)

	with open(op.join(curFolder, 'bbox.txt'), 'w') as out_file:
		print(" ".join(['{:.5f}'.format(f) for f in bbox + [voxel_size]]), file=out_file)

	with open(op.join(curFolder, 'transforms.json'), 'w') as out_file:
		json.dump(json_data, out_file, indent=4)