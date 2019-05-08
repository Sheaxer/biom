#<!python>

import cv2
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil


# https://gist.github.com/kendricktan/93f0da88d0b25087d751ed2244cf770c

def process(img, filters):              #https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/
 accum = np.zeros_like(img)
 for kern in filters:
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
 np.maximum(accum, fimg, accum)
 return accum

def z3b():
    images = []
    for root, dirs, files in os.walk('F://DB/eyes/output/ot/imgs'):
        for file in files:
            p=os.path.join(root,file)
            images.append(p)

    tmpName = 0
    GaborFilters = []
    # Gabor = cv2.getGaborKernel((50, 50), 3.7, -np.pi, 7, 0.5, 75.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)
    # Gabor = cv2.getGaborKernel((50, 50), 3.7, -np.pi, 7, 0.5, 75.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)
    # Gabor = cv2.getGaborKernel((50, 50), 3.7, -np.pi, 7, 0.5, 75.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)
    # Gabor = cv2.getGaborKernel((50, 50), 3.7, -np.pi, 7, 0.5, 30.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)
    # Gabor = cv2.getGaborKernel((50, 50), 3.7, -np.pi, 7, 0.5, 75.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)

    for theta in np.arange(0, np.pi, np.pi / 16):
        Gabor = cv2.getGaborKernel((50, 50), 3.7, theta, 7, 0.5, 75.0, ktype=cv2.CV_32F)
        GaborFilters.append(Gabor)

    for image in images:
        name = image[-11:]
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        #for g in GaborFilters:
        # img1 = cv2.filter2D(img, cv2.CV_8UC3, Gabor)
        # cv2.imshow('filter',img1)
        # cv2.waitKey(0)
            #g /= 1.5 * g.sum()
        img1 = process(img, GaborFilters)
        #img1 = cv2.filter2D(img, cv2.CV_8UC3, g)

        new_img1 = Image.fromarray(img1)
        filename = 'F://DB/eyes/output/ot1/filter/' + name[:7] + '/'
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        new_img1.save('F://DB/eyes/output/ot1/filter/' + name[:7] + '/' + name[:7] + str(tmpName) + name[-4:])

        tmp = np.sum(img1)
        tmp1 = np.size(img1)
        tmp = tmp/tmp1
    #print(tmp)
        ret, thresh1 = cv2.threshold(img1, int(tmp), 255, cv2.THRESH_BINARY)     #int(tmp)
    # cv2.imshow('filter',thresh1)
        # cv2.waitKey(0)
        new_img = Image.fromarray(thresh1)
        filename = 'F://DB/eyes/output/ot1/binary/' + name[:7] + '/'
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        new_img.save('F://DB/eyes/output/ot1/binary/' + name[:7] + '/' + name[:7] + str(tmpName) + name[-4:])
        tmpName += 1

        tmpName = 0


def z3():
    images = []
    for root, dirs, files in os.walk('F://DB/eyes/output/ot/imgs'):
        for file in files:
            p=os.path.join(root,file)
            images.append(p)


    GaborFilters = []
    tmpName = 0
    # https://stackoverflow.com/questions/30071474/opencv-getgaborkernel-parameters-for-filter-bank
    # getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
    # for j in range(1,7):
    #     if j != 5:
    #         # for i in range(0, 10):
    #         #     Gabor = cv2.getGaborKernel((5, 5), 1.0 + 2 * i, np.pi / j, 1.0, 0.5, 30.0, ktype=cv2.CV_32F)
    #         #     GaborFilters.append(Gabor)  # 45 stupnov nemenit?
    #         # for i in range(0, 10):
    #         #     Gabor = cv2.getGaborKernel((5, 5), 1.0, np.pi / j, 1.0 + 2 * i, 0.5, 30.0, ktype=cv2.CV_32F)
    #         #     GaborFilters.append(Gabor)  # 45 stupnov nemenit?
    #         # for i in range(0, 20):
    #         #     Gabor = cv2.getGaborKernel((5, 5), 1.0 + 2 * i, -np.pi / j, 1.0, 0.5, 30.0, ktype=cv2.CV_32F)
    #         #     GaborFilters.append(Gabor)
    #         Gabor = cv2.getGaborKernel((5, 5), 3.5, -np.pi / j, 7, 0.5, 30.0, ktype=cv2.CV_32F)
    #         GaborFilters.append(Gabor)

    # Gabor = cv2.getGaborKernel((5, 5), 3.7, -np.pi, 7, 0.5, 30.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)
    # Gabor = cv2.getGaborKernel((5, 5), 1, -np.pi, 1, 0.5, 30.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)
    # Gabor = cv2.getGaborKernel((5, 5), 7, -np.pi, 10, 0.5, 30.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)
    # Gabor = cv2.getGaborKernel((5, 5), 0.1, -np.pi, 0.1, 0.5, 30.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)
    # Gabor = cv2.getGaborKernel((5, 5), 10, -np.pi, 20, 0.5, 30.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)

    Gabor = cv2.getGaborKernel((50, 50), 3.7, -np.pi, 7, 0.5, 75.0, ktype=cv2.CV_32F)
    GaborFilters.append(Gabor)
    # Gabor = cv2.getGaborKernel((50, 50), 3, -np.pi, 7, 0.5, 75.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)
    # Gabor = cv2.getGaborKernel((50, 50), 3, -np.pi, 6, 1, 75.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)
    # Gabor = cv2.getGaborKernel((5, 5), 3.7, -np.pi, 7, 0.5, 30.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)
    # Gabor = cv2.getGaborKernel((50, 50), 3.7, -np.pi, 3.7, 0.5, 75.0, ktype=cv2.CV_32F)
    # GaborFilters.append(Gabor)

    #
    # for G in GaborFilters:
    #     # GaborShow = cv2.resize(Gabor,(100, 100), interpolation = cv2.INTER_LINEAR)
    #     # cv2.imshow('kernel', GaborShow)
    #     # cv2.waitKey(0)
    #     G /= 1.5 * G.sum()
    #     # GaborShow1 = cv2.resize(Gabor, (100, 100), interpolation=cv2.INTER_LINEAR)
    #     # cv2.imshow('kernel', GaborShow1)
    #     # cv2.waitKey(0)

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
            filename = 'F://DB/eyes/output/ot/filter/' + name[:7] + '/'
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            new_img1.save('F://DB/eyes/output/ot/filter/' + name[:7] + '/' + name[:7] + str(tmpName) + name[-4:])

            tmp = np.sum(img1)
            tmp1 = np.size(img1)
            tmp = tmp/tmp1
        #print(tmp)
            ret, thresh1 = cv2.threshold(img1, int(tmp), 255, cv2.THRESH_BINARY)     #int(tmp)
        # cv2.imshow('filter',thresh1)
        # cv2.waitKey(0)
            new_img = Image.fromarray(thresh1)
            filename = 'F://DB/eyes/output/ot/binary/' + name[:7] + '/'
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            new_img.save('F://DB/eyes/output/ot/binary/' + name[:7] + '/' + name[:7] + str(tmpName) + name[-4:])
            tmpName += 1

        tmpName = 0


def z3_porovnanie():
    images = []
    imgs = []
    keys = []
    values = []
    same_eye = []
    name = []
    for root, dirs, files in os.walk('F://DB/eyes/output/ot/binary/'):
        for file in files:
            p = os.path.join(root, file)
            images.append(p)

    for image in images:
        name.append(image[-12:])
        #imgs.append(cv2.imread(name))
        if name[len(name)-1][7] == '0':
            #imgs[name] = cv2.imread(image)
            #imgs.update({name:cv2.imread(image)})
            imgs.append(image)

    # for x,y in imgs.items():
    #     keys.append(x)
    #     values.append(y)

    tmp = 0
    for i in range(1,11):
        for im in imgs:
            if im[28] == str(i) and tmp<2:
                filename = 'F://DB/eyes/output/ot/same_eye_new/' + str(i) + '/'
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))
                shutil.copy(im,filename)
                tmp += 1
        tmp = 0

def z3_porovnanie1():
    images = []
    imgs = []
    same_eye = []
    name = []
    for root, dirs, files in os.walk('F://DB/eyes/output/ot/binary/'):
        for file in files:
            p = os.path.join(root, file)
            images.append(p)

    for image in images:
        name.append(image[-12:])
        # imgs.append(cv2.imread(name))
        if name[len(name) - 1][7] == '0':
            # imgs[name] = cv2.imread(image)
            # imgs.update({name:cv2.imread(image)})
            imgs.append(image)

    # for x,y in imgs.items():
    #     keys.append(x)
    #     values.append(y)

    tmp = 0
    for i in range(1, 11):
        for im in imgs:
            if im[28] == str(i) and tmp < 1:
                filename = 'F://DB/eyes/output/ot/diff_eye_new/' + str(i) + '/'
                im2 = list(im)
                if i<10:
                    im2[28] = str(i+1)
                    im2[36] = str(i+1)
                else:
                    im2[28] = '1'
                    im2[36] = '1'
                im3 = ''.join(im2)
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))
                shutil.copy(im, filename)
                shutil.copy(im3, filename)
                tmp += 1
            elif tmp == 1:
                break
        tmp = 0

# same_eye.append(imgs[name[1]])
# same_eye.append(imgs[name[0]])

# f, axarr = plt.subplots(1, 2)
# axarr[0, 0].imshow(same_eye[0])
# axarr[0, 1].imshow(same_eye[1])


z3()
# z3b()
#z3_porovnanie()
# z3_porovnanie1()