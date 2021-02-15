import sys, os, argparse
import numpy as np

# print("Python version")
# print (sys.version)
# print (sys.executable)
# print("Version info.")
# print (sys.version_info)
# exit()

import imageio
import OpenEXR, Imath

print ('\n########## POSTPROCESSING ##########')
print (sys.argv)

parser = argparse.ArgumentParser(description='Postprocesses the created dataset')
parser.add_argument('images_folder', metavar='imgs', type=str, help='path where files has been saved')
parser.add_argument('output_folder', metavar='out', type=str, help='path where results should be saved')

argv = sys.argv[1:]
args = parser.parse_args(argv)

mu = None
mus = None
i = 0
png = True

print ('## Reading files: ')
for flnm in os.listdir(args.images_folder):
    flpth = os.path.join(args.images_folder, flnm)
    print (flpth)
    if flnm.endswith(".png"): 
        img = imageio.imread(flpth)
    if OpenEXR.isOpenExrFile(flpth):
        png = False
        exr = OpenEXR.InputFile(flpth)
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
        if 'A' in ch:
            a = np.frombuffer(exr.channel('A'), dtype=tps[ch['A'].type.v])
            img = np.stack((r, g, b, a)).reshape(4, sz[0]*sz[1]).T
        else:
            img = np.stack((r, g, b)).reshape(3, sz[0]*sz[1]).T

    if mu is None: mu = img
    else: mu = (mu * i + img) / (i + 1)

    if mus is None: mus = img
    else: mus = (mus * i + np.square(img)) / (i + 1)

    i += 1

print ('## Saving results: ')
if png:
    flpth = os.path.join(args.output_folder, 'mean.png')
    print (flpth)
    imageio.imwrite(flpth, img.astype(np.ubyte))
else:
    flpth = os.path.join(args.output_folder, 'mean.exr')
    print (flpth)
    exr = OpenEXR.OutputFile(flpth, hdr)
    exr.writePixels({'R': img[:,0].reshape(sz[0], -1),
                     'G': img[:,1].reshape(sz[0], -1),
                     'B': img[:,2].reshape(sz[0], -1),
                     'A': img[:,3].reshape(sz[0], -1)})