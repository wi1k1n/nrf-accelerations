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
print ('argv:', sys.argv)

# print('Not Implemented!')
# exit()

# CALCULATE MEAN and STD
parser = argparse.ArgumentParser(description='Postprocesses the created dataset')
parser.add_argument('images_folder', metavar='imgs', type=str, help='path where files has been saved')
parser.add_argument('output_folder', metavar='out', type=str, help='path where results should be saved')
parser.add_argument('percentile_min', metavar='percentile_min', type=float, help='low percentile', default=0.5)
parser.add_argument('percentile_max', metavar='percentile_max', type=float, help='high percentile', default=99.5)

argv = sys.argv[1:]
args = parser.parse_args(argv)

mu = []
muxs = []
mins, maxs = [], []
prcn, prcx = [], []
i = 0
png = True

print ('## Reading files: ')
flnmlen = len(os.listdir(args.images_folder)) - 1
# images = np.empty((flnmlen + 1, 1024, 1024, 4))
for flnm in os.listdir(args.images_folder):
    flpth = os.path.join(args.images_folder, flnm)
    print('    [{}/{}] '.format(i, flnmlen), flpth)
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

    img = img.astype(np.float32)

    # images[i,...] = img.reshape(sz[0], sz[1], -1)

    cmean = img.mean(0)
    xsmean = np.square(img).mean(0)
    # # print('type:', img.dtype)
    # print('min:', img.min(0))
    # print('max:', img.max(0))
    # print('E[x]:', cmean)
    # print('E[x^2]:', cstd)
    mu.append(cmean)
    muxs.append(xsmean)
    mins.append(img.min(0))
    maxs.append(img.max(0))
    prcn.append(np.percentile(img, args.percentile_min, axis=0))
    prcx.append(np.percentile(img, args.percentile_max, axis=0))

    # if mu is None: mu = cmean
    # else: mu = (mu * i + cmean) / (i + 1)

    # if mus is None: mus = np.square(cmean)
    # else: mus = (mus * i + np.square(cmean)) / (i + 1)

    i += 1

print ('{0} samples in dataset'.format(i))

if i:
    mu_g = np.array(mu).mean(0)  # E[x] for the whole dataset
    muxs_g = np.array(muxs).mean(0)  # E(x^2) for the whole dataset
    std_g = np.sqrt(muxs_g - np.square(mu_g))  # std_g for the whole dataset
    min_g = np.array(mins).min(0)
    max_g = np.array(maxs).max(0)
    prcn_g = np.median(np.array(prcn), axis=0)
    prcx_g = np.median(np.array(prcx), axis=0)

    # imgs = images.reshape(-1, images.shape[-1])
    # mu_g = imgs.mean(0)
    # std_g = imgs.std(0)
    # min_g = imgs.min(0)
    # max_g = imgs.max(0)
    # prcn_g = np.percentile(imgs, args.percentile_min, axis=0)
    # prcx_g = np.percentile(imgs, args.percentile_max, axis=0)

    print('Mean: {}'.format(mu_g))
    print('Std: {}'.format(std_g))
    print('Min: {}'.format(min_g))
    print('Max: {}'.format(max_g))
    print('PrcnMin: {}'.format(prcn_g))
    print('PrcnMax: {}'.format(prcx_g))

    print ('## Saving postprocessing data: ')
    with open(os.path.join(args.output_folder, OUTPUTFILE), 'w') as fi:
        print(', '.join(map(str, list(mu_g))), file=fi)
        print(', '.join(map(str, list(std_g))), file=fi)
        print(', '.join(map(str, list(min_g))), file=fi)
        print(', '.join(map(str, list(max_g))), file=fi)
        print(', '.join(map(str, list(prcn_g))), file=fi)
        print(', '.join(map(str, list(prcx_g))), file=fi)
else:
    print ('## No samples found. No data has been saved')