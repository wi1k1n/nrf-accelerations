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

OUTPUTFILE = 'postprocessing.txt'

print ('\n########## POSTPROCESSING ##########')
print (sys.argv)

# print('Not Implemented!')
# exit()

# CALCULATE MEAN and STD
parser = argparse.ArgumentParser(description='Postprocesses the created dataset')
parser.add_argument('images_folder', metavar='imgs', type=str, help='path where files has been saved')
parser.add_argument('output_folder', metavar='out', type=str, help='path where results should be saved')

argv = sys.argv[1:]
args = parser.parse_args(argv)

mu = []
mus = []
i = 0
png = True

print ('## Reading files: ')
for flnm in os.listdir(args.images_folder):
    flpth = os.path.join(args.images_folder, flnm)
    print (flpth)
    if flnm.endswith(".png"): 
        img = imageio.imread(flpth)
        img = img.reshape(-1, img.shape[-1])
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

    cmean = img.mean(0)
    mu.append(cmean)
    mus.append(np.square(cmean))

    # if mu is None: mu = cmean
    # else: mu = (mu * i + cmean) / (i + 1)

    # if mus is None: mus = np.square(cmean)
    # else: mus = (mus * i + np.square(cmean)) / (i + 1)

    i += 1

print ('{0} samples in dataset'.format(i))

mu_g = np.array(mu).mean(0)  # mu for the whole dataset
mus_g = np.array(mus).mean(0)  # mu^2 for the whole dataset
std_g = np.sqrt(mus_g - np.square(mu_g))  # std_g for the whole dataset

print('Mean: {}'.format(mu_g))
print('Std: {}'.format(std_g))

print ('## Saving postprocessing data: ')
with open(os.path.join(args.output_folder, OUTPUTFILE), 'w') as fi:
    print(', '.join(map(str, list(mu_g))), file=fi)
    print(', '.join(map(str, list(std_g))), file=fi)