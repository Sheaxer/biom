#<!python>

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import subprocess
import z5_3
import z5
import z5_2
import csv
import numpy as np
import csv
import cv2
import scipy as sp
import scipy.ndimage
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
import shutil

def fnc(imagePath):
    import cv2
    image = cv2.imread(imagePath)
    name = imagePath[-11:]

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y



def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2pol(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nx)
    theta_i = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = pol2cart(r_grid, theta_grid)
    xi += origin[0] # We need to shift the origin back to
    yi += origin[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)
    bands = []
    for band in data.T:
        zi = sp.ndimage.map_coordinates(band, coords, order=1)
        bands.append(zi.reshape((nx, ny)))
    output = np.dstack(bands)
    return output, r_i, theta_i

def resizeHelper(image):
    imageNew = np.zeros((493, 170,3))
    for f in range(image.shape[0]):
        tmp = 0
        tmpRow = np.zeros((493, 180,3))
        for g in range(image.shape[1]):
            #if (image[f,g][0] != 41) & (image[f,g][1] != 41) & (image[f,g][2] != 41):
            if (image[f,g][0] > 65) & (image[f,g][1] > 65) & (image[f,g][2] > 65) | (g<85):
                tmpRow[0][tmp] = image[f,g]
                tmp += 1
            if tmp == 170:
                imageNew[f]= np.array(tmpRow[0][:tmp])
            else:
                imageNew[f] = cv2.resize(np.array(tmpRow[0][:tmp]),(1,170), interpolation=cv2.INTER_CUBIC)
                continue
    imageNew = np.round(imageNew[:, :]).astype("uint8")
    #cv2.imshow('aa', image)
    #cv2.imshow('imNew', imageNew)
    #cv2.waitKey(0)
    return imageNew
def resizeHelperMask(image):
    imageNew = np.zeros((493, 170,3))
    #imageNew = imageNew.fill(0)
    for f in range(image.shape[0]):
        tmp = 0
        tmpRow = np.zeros((493, 180,3))
        for g in range(image.shape[1]):
            #if (image[f,g][0] != 41) & (image[f,g][1] != 41) & (image[f,g][2] != 41):
            if (image[f,g][0] == 0 ) & (image[f,g][1] == 0) & (image[f,g][2] == 0) | (g<110):
                tmpRow[0][tmp] = image[f,g]
                tmp+=1
                continue
            if tmp == 170:
                imageNew[f]= np.array(tmpRow[0][:tmp])
                continue
            else:
                #imageNew[f][:tmp] = np.array(tmpRow[0][:tmp])
                imageNew[f] = cv2.resize(np.array(tmpRow[0][:tmp]),(1,170), interpolation=cv2.INTER_NEAREST )
                continue

    # for f in range(image.shape[0]):
    #     for g in range(image.shape[1]):
    #         if (image[f,g][0] == 0 ) & (image[f,g][1] == 0) & (image[f,g][2] == 0) | (g<110):
    imageNew = np.round(imageNew[:, :]).astype("uint8")
    #cv2.imshow('aa', image)
    #cv2.imshow('imNew', imageNew)
    #cv2.waitKey(0)
    return imageNew
def plot_polar_image(data, r1, r2, name, origin=None):
    """Plots an image reprojected into polar coordinages with the origin
    at "origin" (a tuple of (x0, y0), defaults to the center of the image)"""
    polar_grid, r, theta = reproject_image_into_polar(data, origin)
    plt.figure()
    #polar_grid = polar_grid[int(r1):int(r2),:]
    plt.imshow(polar_grid, extent=(theta.min(), theta.max(), r.max(), r.min()))
    plt.axis('auto')
    plt.ylim(plt.ylim()[::-1])
    #plt.xlabel('Theta Coordinate (radians)')
    #plt.ylabel('R Coordinate (pixels)')
    #plt.title('Image in Polar Coordinates')
    plt.savefig('testplot.png')
    image = cv2.imread('testplot.png')
    image = image[70+int(r1):239+int(r1),82:575]
    image = np.rot90(image)
    from PIL import Image
    #im = Image.fromarray(image)
    #im.save('C://Users/vizva/OneDrive/Dokumenty/GitHub/biometria/outputUnwrap/unwrap' +name)
    im = resizeHelper(image)
    im = np.rot90(im)
    im = np.rot90(im)
    im = np.rot90(im)
    # cv2.imshow('bruh',im)
    #     # cv2.waitKey(0)
    im=Image.fromarray(im)
    im.save('F://DB/eyes/vystup_test/ot/imgs/' +name.replace("\\",""))

    plt.show(block=False)
    #plt.show()
    plt.clf()
    plt.close()
def plot_polar_imageMask(data, r1, r2, name, origin=None):
    """Plots an image reprojected into polar coordinages with the origin
    at "origin" (a tuple of (x0, y0), defaults to the center of the image)"""
    polar_grid, r, theta = reproject_image_into_polar(data, origin)
    plt.figure()
    #polar_grid = polar_grid[int(r1):int(r2),:]
    plt.imshow(polar_grid, extent=(theta.min(), theta.max(), r.max(), r.min()))
    plt.axis('auto')
    plt.ylim(plt.ylim()[::-1])
    #plt.xlabel('Theta Coordinate (radians)')
    #plt.ylabel('R Coordinate (pixels)')
    #plt.title('Image in Polar Coordinates')
    plt.savefig('testplot.png')
    image = cv2.imread('testplot.png')
    image = image[70+int(r1):239+int(r1),82:575]
    image = np.rot90(image)
    from PIL import Image
    #im = Image.fromarray(image)
    #im.save('C://Users/vizva/OneDrive/Dokumenty/GitHub/biometria/outputUnwrap/unwrap' +name)
    im = resizeHelperMask(image)
    im = np.rot90(im)
    im = np.rot90(im)
    im = np.rot90(im)
    im=Image.fromarray(im)
    im.save('F://DB/eyes/vystup_test/ot/M/' + name.replace("\\",""))

    plt.show(block=False)
    plt.clf()
    plt.close()

def func(obrazok):
    with open('F://DB/eyes/iris_bounding_circles.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[0].find(obrazok[-11:])>(-1):
                circle = {'name': row[0], 'x1': int(row[1]), 'y1': int(row[2]), 'r1': float(row[3]), 'x2': int(row[4]),
                          'y2': int(row[5]), 'r2': float(row[6]), 'x3': int(row[7]), 'y3': int(row[8]),
                          'r3': float(row[9]), 'x4': int(row[10]), 'y4': int(row[11]), 'r4': float(row[12])}
                circleMiddleBoth = {'name': row[0], 'x1': int(row[1]), 'y1': int(row[2]), 'r1': float(row[3]),
                                    'x2': int(row[4]), 'y2': int(row[5]), 'r2': float(row[6])}
                circleMiddleBig = {'name': row[0], 'x2': int(row[4]), 'y2': int(row[5]), 'r2': float(row[6])}
                circleLOWER = {'name': row[0], 'x': int(row[7]), 'y': int(row[8]), 'r': float(row[9])}
                circleUPPER = {'name': row[0], 'x': int(row[10]), 'y': int(row[11]), 'r': float(row[12])}

                rectX = int(circleMiddleBig['x2'] - circleMiddleBig['r2'])
                rectY = int(circleMiddleBig['y2'] - circleMiddleBig['r2'])
                image = cv2.imread(obrazok)
                name = obrazok[-11:]

                bordersize = 100
                image = cv2.copyMakeBorder(image, top=bordersize, bottom=bordersize, left=0, right=0,
                                           borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
                image = image.copy()

                mask = np.zeros((image.shape), np.uint8)
                mask2 = np.zeros((image.shape), np.uint8)
                mask3 = np.zeros((image.shape), np.uint8)
                mask4 = np.zeros((image.shape), np.uint8)
                cv2.circle(mask, (int(circleMiddleBig['x2']), int(circleMiddleBig['y2'])), int(circleMiddleBig['r2']),
                           (255, 255, 255), thickness=-1)
                cv2.circle(mask2, (int(circleMiddleBoth['x1']), int(circleMiddleBoth['y1'])),
                           int(circleMiddleBoth['r1']), (255, 255, 255), thickness=-1)

                cv2.circle(mask3, (int(circleLOWER['x']), int(circleLOWER['y'])), int(circleLOWER['r']),
                           (255, 255, 255), thickness=-1)
                cv2.circle(mask4, (int(circleUPPER['x']), int(circleUPPER['y'])), int(circleUPPER['r']),
                           (255, 255, 255), thickness=-1)

                # duhovka        #dole              #hore            #stred
                mask_inv = (mask == 255)  # & (mask3 == 255) #& (mask4 == 255) #& (mask2 == 0)
                masked = image * mask_inv
                maska = mask_inv & (mask3 == 255) & (mask4 == 255) & (mask2 == 0)  # & (image == 255)

                maska = image * maska

                from PIL import Image

                img = maska
                # cv2.imshow('img',img)
                # cv2.waitKey(0)
                maskTMP = np.zeros((image.shape), np.uint8)
                maskTMP.fill(255)
                for f in range(img.shape[0]):  # for every pixel:
                    for g in range(img.shape[1]):
                        if img[f, g][0] == 0 & img[f, g][1] == 0 & img[f, g][2] == 0:  # if not black:
                            img[f, g][0] = 255
                            img[f, g][1] = 255
                            img[f, g][2] = 255  # change to white
                        else:
                            img[f, g][0] = 0
                            img[f, g][1] = 0
                            img[f, g][2] = 0

                # cv2.imshow('maska',img)
                # out = cv2.linearPolar(masked, masked, (masked.shape[0]/2,masked.shape[1]/2),circleMiddleBoth['r2'], cv2.WARP_FILL_OUTLIERS)
                # cv2.imshow('a',out)
                # cv2.waitKey(0)
                #j += 1
                crop_img = masked[int(circleMiddleBig['y2'] - circleMiddleBig['r2']):(
                    int(circleMiddleBig['y2'] + circleMiddleBig['r2'])),
                           int(circleMiddleBig['x2'] - circleMiddleBig['r2']):(
                               int(circleMiddleBig['x2'] + circleMiddleBig['r2']))]
                # print(circle)
                crop_img_mask = img[int(circleMiddleBig['y2'] - circleMiddleBig['r2']):(
                    int(circleMiddleBig['y2'] + circleMiddleBig['r2'])),
                                int(circleMiddleBig['x2'] - circleMiddleBig['r2']):(
                                    int(circleMiddleBig['x2'] + circleMiddleBig['r2']))]

                # cv2.imshow('output', crop_img)
                # cv2.imshow('output2', crop_img_mask)
                # cv2.waitKey(0)

                plot_polar_image(crop_img, circleMiddleBoth['r2'], circleMiddleBoth['r1'], name, origin=None)
                plot_polar_imageMask(crop_img_mask, circleMiddleBoth['r2'], circleMiddleBoth['r1'], name, origin=None)
                print(name + " done")
                # cv2.waitKey(0)

def z3():
    images = []
    for root, dirs, files in os.walk('F://DB/eyes/vystup_test/ot/imgs'):
        for file in files:
            p=os.path.join(root,file)
            images.append(p)


    GaborFilters = []
    tmpName = 0

    Gabor = cv2.getGaborKernel((50, 50), 3.7, -np.pi, 7, 0.5, 75.0, ktype=cv2.CV_32F)
    GaborFilters.append(Gabor)


    for image in images:
        name = image[-11:]
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        for g in GaborFilters:
        # img1 = cv2.filter2D(img, cv2.CV_8UC3, Gabor)
        # cv2.imshow('filter',img1)
        # cv2.waitKey(0)
        #    g /= 1.5 * g.sum()
            img1 = cv2.filter2D(img, cv2.CV_8UC3, g)

            new_img1 = Image.fromarray(img1)
            filename = 'F://DB/eyes/vystup_test/ot/filter/' + name[:7] + '/'
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            new_img1.save('F://DB/eyes/vystup_test/ot/filter/' + name[:7] + '/' + name[:7] + str(tmpName) + name[-4:])

            tmp = np.sum(img1)
            tmp1 = np.size(img1)
            tmp = tmp/tmp1
        #print(tmp)
            ret, thresh1 = cv2.threshold(img1, int(tmp), 255, cv2.THRESH_BINARY)     #int(tmp)
        # cv2.imshow('filter',thresh1)
        # cv2.waitKey(0)
            new_img = Image.fromarray(thresh1)
            filename = 'F://DB/eyes/vystup_test/ot/binary/' + name[:7] + '/'
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            new_img.save('F://DB/eyes/vystup_test/ot/binary/' + name[:7] + '/' + name[:7] + str(tmpName) + name[-4:])
            tmpName += 1

        tmpName = 0


def predict(neuronka, mi, sigma):
    images = []
    masky = []
    oko = 0
    tmp = []
    tmp2 = []
    prve_cislo = 1
    druhe_cislo = 1
    oci = {}
    oci2 = {}
    x = 97
    ctr = 0
    for root, dirs, files in os.walk('F://DB/eyes/vystup_test/ot/M'):
        for file in files:
            p = os.path.join(root, file)
            masky.append(p)

    # while prve_cislo < 10:
    #     for root, dirs, files in os.walk('F://BIOM/z4/same_eye/00' + str(prve_cislo) + '_' + str(druhe_cislo)):
    #         for file in files:
    #             p = os.path.join(root, file)
    #             tmp.append(p)
    #         oci[str(prve_cislo) + '_' + str(druhe_cislo)] = tmp.copy()
    #         tmp = []
    #         if druhe_cislo == 1:
    #             druhe_cislo = druhe_cislo + 1
    #         else:
    #             druhe_cislo = 1
    #         if druhe_cislo == 1:
    #             prve_cislo = prve_cislo + 1

    for root, dirs, files in os.walk('F://DB/eyes/vystup_test/ot/binary'):
        for file in files:
            p = os.path.join(root, file)
            if p.endswith(('.bmp')):
                tmp.append(p)

    while x < 118:
        for pic in tmp:
            if (chr(x) == pic[-10]):
                tmp2.append(pic)
        oci[str('00' + str(chr(x)) + '_1')] = tmp2.copy()
        x = x + 1
        tmp2 = []

    oci2.fromkeys(oci.keys(), [])
    tmp = []
    for x, a in oci.items():
        for b in a:
            tmp1 = b
            b = cv2.imread(b, cv2.IMREAD_GRAYSCALE)
            for m in masky:
                if m[-11:-4] == tmp1[-12:-5]:
                    im_maska = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
                    b = b & np.invert(im_maska)  # & im_maska2
                    tmp.append(b)
        oci2[x] = tmp.copy()
        tmp = []

    # cv2.imshow('im',oci2['00c_1'][0])
    # cv2.waitKey(0)

    list_z = []
    z_score = []

    for k,oko in oci2.items():
        if len(oko)!=0:
            for o in oko:
                o1 = o.reshape(170 * 493)
                guess = neuronka.predict_proba(o1.reshape(1,-1))
                print(str(neuronka.predict(o1.reshape(1,-1))))
                for x in guess:
                    z = (x-mi)/sigma
                    list_z.append(z)
                z_score.append(max(z))

    print(z_score)

if __name__ == "__main__":
    import os
    images = []
    for root, dirs, files in os.walk('F://DB/eyes'):
        for file in files:
            p = os.path.join(root, file)
            if p.endswith(('.bmp')):
                images.append(p)

    z5.fnc(images)
    z5_2.z3()
    mi, sigma, neuronka = z5_3.rovnake_oci2()

    img = "F://DB/eyes/00c/00c_1_2.bmp"
    func(img)
    z3()
    predict(neuronka,mi,sigma)