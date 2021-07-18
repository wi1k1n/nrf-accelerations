import numpy as np, os, os.path as op
import OpenEXR, Imath

PATH = "../blender/_pl_test/render.exr"
CAM_HEIGHT = 5
PL_HEIGHT = 10
PL_POWER = 1000
PLANESZ = (10, 10)

R = 0.1 / 2
Kc, Kl, Kq = 1.0, 2.0 / R, 1.0 / R ** 2
Kd = 0.8


# Reading EXR
path2exr = op.abspath(PATH)
assert OpenEXR.isOpenExrFile(path2exr), 'Provided file is not an EXR file'

exr = OpenEXR.InputFile(path2exr)
hdr = exr.header()
dw = hdr['dataWindow']
ch = hdr['channels']
if not ('R' in ch and 'G' in ch and 'B' in ch):
	raise ValueError('Wrong EXR data')

sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
tps = {Imath.PixelType.UINT: np.uint, Imath.PixelType.HALF: np.half, Imath.PixelType.FLOAT: float}

r = np.frombuffer(exr.channel('R'), dtype=tps[ch['R'].type.v])
g = np.frombuffer(exr.channel('G'), dtype=tps[ch['G'].type.v])
b = np.frombuffer(exr.channel('B'), dtype=tps[ch['B'].type.v])
a = np.frombuffer(exr.channel('A'), dtype=tps[ch['A'].type.v])
img = np.stack((r, g, b, a)).reshape(4, sz[0] * sz[1]).T
img = np.stack((r, g, b, a)).reshape(4, sz[0] * sz[1]).T
img = img.reshape(sz[0], sz[1], -1).astype('float32')


# Managing sizes
cxpx, cypx = sz[0] // 2, sz[1] // 2

imgcum = np.sum(img[...,:3], axis=-1)
imgnzx, imgnzy = np.nonzero(imgcum)
wpx, hpx = imgnzx.max() - imgnzx.min(), imgnzy.max() - imgnzy.min()
shiftxpx, shiftypx = imgnzx.min(), imgnzy.min()
scale = (PLANESZ[0] / wpx, PLANESZ[1] / hpx)

print('Scale: (', scale[0], scale[1], ') m/px')

for i in range(0, hpx):
	for j in range(0, wpx):
		L0 = img[shiftypx + i, shiftxpx + j, :]
		dx = np.linalg.norm([(cxpx - j) * scale[0], (cypx - i) * scale[1]])
		d = np.linalg.norm([dx, PL_HEIGHT])
		costh = float(PL_HEIGHT) / d

		# print('L0 = {}\td = {}'.format(L0, d))
		I = np.pi / Kd * L0 / costh * (Kc + Kl * d + Kq * d ** 2)
		print('px [{}, {}]:\t{}'.format(shiftypx + i, shiftxpx + j, I))