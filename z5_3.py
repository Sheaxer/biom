#<!python>

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

def hamming_dist(obr1, obr2):
    distances = []
    for i in range(0,170):
        # obr1 = cv2.cvtColor(obr1, cv2.COLOR_BGR2GRAY)
        # obr2 = cv2.cvtColor(obr2, cv2.COLOR_BGR2GRAY)
        #obr1 = np.invert(obr1)
        #obr2 = np.invert(obr2)
        # cv2.imshow('im', obr1)
        # cv2.waitKey(0)
        # cv2.imshow('im2', obr2)
        # cv2.waitKey(0)
        # tmp = obr1 & obr2
        # tmp = np.invert(tmp)
        # cv2.imshow('im3', tmp)
        # cv2.waitKey(0)
        distances.append(np.count_nonzero(obr1 != obr2))
        #print(np.count_nonzero(obr1 != obr2))
        obr1 = np.roll(obr1,1)
        #obr2 = np.roll(obr2,1)
    return min(distances)

def z3_same_eye():
    images = []
    imgs = []
    name = []
    masky = []
    vzdialenosti = []
    counter = 0
    for root, dirs, files in os.walk('F://BIOM/z4/same_eye'):
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

    for root, dirs, files in os.walk('F://BIOM/output/ot/M'):
        for file in files:
            p = os.path.join(root, file)
            masky.append(p)

    for i in images:
        im = cv2.imread(i)
        counter = counter + 1
        for m in masky:
            if m[21:28] == i[27:34]:
                im_maska = cv2.imread(m)
                #im_maska = np.invert(im_maska)
        # im = im & im_maska
        # cv2.imshow('im',im)
        # cv2.waitKey(0)
        for j in range(counter,len(images)):
            if i[21:26] == images[j][21:26]:
                im2 = cv2.imread(images[j])
                for m in masky:
                    if m[21:28] == images[j][27:34]:
                        im_maska2 = cv2.imread(m)
                        #im_maska2 = np.invert(im_maska2)
                #print(i + "  -  " + images[j])
                im = im & im_maska #& im_maska2
                im2 = im2 & im_maska2 #& im_maska
                #print(hamming_dist(im, im2))
                vzdialenosti.append(hamming_dist(im, im2))

    priemer = sum(vzdialenosti)/len(vzdialenosti)
    print("Priemerna vzdialenost je " + str(priemer))
    #print("Prvkov je " + str(len(vzdialenosti)))
    vz_original = vzdialenosti.copy()
    vzdialenosti.sort()
    print("Najmensia vzdialenost je " + str(vzdialenosti[0]))
    print("Najvacsia vzdialenost je " + str(vzdialenosti[len(vzdialenosti)-1]))

    return vzdialenosti

def z3_diff_eye():
    images = []
    imgs = []
    name = []
    masky = []
    vzdialenosti = []
    counter = 0

    for root, dirs, files in os.walk('F://BIOM/output/ot/M'):
        for file in files:
            p = os.path.join(root, file)
            masky.append(p)
    for ss in range(1,3):
        for root, dirs, files in os.walk('F://BIOM/z4/diff_eye/' + str(ss)):
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

            for i in images:
                im = cv2.imread(i)
                counter = counter + 1
                for m in masky:
                    if m[21:28] == i[23:30]:
                        im_maska = cv2.imread(m)
                        #im_maska = np.invert(im_maska)
                # im = im & im_maska
                # cv2.imshow('im',im)
                # cv2.waitKey(0)
                for j in range(counter,len(images)):
                    im2 = cv2.imread(images[j])
                    for m in masky:
                        if m[21:28] == images[j][23:30]:
                            im_maska2 = cv2.imread(m)
                            #im_maska2 = np.invert(im_maska2)
                    #print(i + "  -  " + images[j])
                    im = im & im_maska #& im_maska2
                    im2 = im2 & im_maska2 #& im_maska1
                    #print(hamming_dist(im, im2))
                    vzdialenosti.append(hamming_dist(im, im2))

            images = []
            imgs = []
            name = []
            counter = 0



    priemer = sum(vzdialenosti)/len(vzdialenosti)
    print("Priemerna vzdialenost je " + str(priemer))
    #print("Prvkov je " + str(len(vzdialenosti)))
    vz_original = vzdialenosti.copy()
    vzdialenosti.sort()
    print("Najmensia vzdialenost je " + str(vzdialenosti[0]))
    print("Najvacsia vzdialenost je " + str(vzdialenosti[len(vzdialenosti)-1]))

    return vzdialenosti


def hamming_dist_rotacia(obr1, obr2):
    distances = []
    for i in range(0, 170):
        distances.append(np.count_nonzero(obr1 != obr2))
        obr1 = np.roll(obr1, 1)
    min_hamming = min(distances)
    return distances.index(min_hamming)


def rovnake_oci2():
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
    for root, dirs, files in os.walk('F://DB/eyes/output/ot/M'):
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

    for root, dirs, files in os.walk('F://DB/eyes/output/ot/binary'):
        for file in files:
            p = os.path.join(root, file)
            if p.endswith(('.bmp')):
                tmp.append(p)

    while x < 118:
        for pic in tmp:
            if(chr(x) == pic[-10]):
                tmp2.append(pic)
        oci[str('00' + str(chr(x)) + '_1')] = tmp2.copy()
        x = x + 1
        tmp2 = []

    oci2.fromkeys(oci.keys(),[])
    tmp = []
    for x,a in oci.items():
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


    for kluc, hodnota in oci2.items():
        for i in range(1,len(hodnota)):
            posun = hamming_dist_rotacia(hodnota[0], hodnota[i])
            for j in range(0, posun):
                hodnota[i] = np.roll(hodnota[i],1)
            oci2[kluc][i] = hodnota[i]


    listOci = []
    labels = []
    for kluc in oci2.keys():
        for oko in oci2[kluc]:
            labels.append(kluc)
            listOci.append(oko)

    c = list(zip(listOci, labels))
    import random
    random.shuffle(c)

    listOci, labels = zip(*c)

    from sklearn.neural_network import MLPClassifier


    trainList = []
    trainLab = []
    testList = []
    testLab = []

    for i in range(0,60):
        trainList.append(listOci[i])
        trainLab.append(labels[i])
    for i in range(50,len(masky)):
        testList.append(listOci[i])
        testLab.append(labels[i])

    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(100,100,),
                        learning_rate='adaptive', learning_rate_init=0.01,
                        max_iter=200, momentum=0.1, n_iter_no_change=20,        #0.01 0,01 ... 0.49 ...         = 1
                        nesterovs_momentum=True, power_t=0.5, random_state=1,
                        shuffle=True, solver='adam', tol=0.0001,
                        validation_fraction=0.1, verbose=False, warm_start=False)
    #trainList = np.reshape(trainList, (100, 28)).T
    trainList1 = []
    for k in trainList:
        trainList1.append(k.reshape(170 * 493))


    testList1 = []
    for k in testList:
        testList1.append(k.reshape(170 * 493))

    clf.fit(trainList1, trainLab)
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    # clf = pickle.load(open(filename, 'rb'))
    print("Uspesnost je " + str(clf.score(testList1,testLab)))

    vysledky_z_mlp = clf.predict_proba(trainList1)

    file123 = open('Vysledky_z_MLP_train.txt', 'a')
    for line in vysledky_z_mlp:
        file123.write(str(line)+'\n')
    file123.close()

    valTPR = []
    valFPR = []
    #hranica = [0.01, 0.02, 0.03, 0.04, 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    hranica = [0.01, 0.02, 0.03, 0.04, 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
    ctr = 0
    ctr1 = 0

    mi = np.mean(vysledky_z_mlp)
    delta = np.std(vysledky_z_mlp)
    print('mi = ' + str(mi) + ', delta = ' + str(delta))
    for i in range(0, vysledky_z_mlp.shape[0]):
        for j in range(0, vysledky_z_mlp.shape[1]):
            vysledky_z_mlp[i][j] = ((vysledky_z_mlp[i][j]) - mi) / delta

    #print(vysledky_z_mlp.shape)
    for value_for_TPR in vysledky_z_mlp:
        label_oka = trainLab[ctr][2]
        cislo_v_riadku = ord(label_oka) - 97
        valTPR.append(value_for_TPR[cislo_v_riadku])
        ctr = ctr + 1

    for value_for_FPR in vysledky_z_mlp:
        label_oka = trainLab[ctr1][2]
        cislo_v_riadku = ord(label_oka) - 97
        for prvok in value_for_FPR:
            if prvok == value_for_FPR[cislo_v_riadku]:
                continue
            else:
                valFPR.append(prvok)
        ctr1 = ctr1 + 1

    z_scoreTPR = []
    z_scoreFPR = []

    for h in hranica:
        ctr = 0
        for v in valTPR:
            if v > h:
                ctr = ctr + 1
        tmp_Z_Score = ctr/len(valTPR)
        z_scoreTPR.append(tmp_Z_Score)

    for h in hranica:
        ctr = 0
        for v in valFPR:
            if v > h:
                ctr = ctr + 1
        tmp_Z_Score = ctr/len(valFPR)
        z_scoreFPR.append(tmp_Z_Score)

    # for i in range(0,len(z_scoreTPR)):
    #     print('TPR = ' + str(z_scoreTPR[i]) + ', FPR = ' + str(z_scoreFPR[i]))

    #print(str(z_score))
    plt.plot(z_scoreFPR,z_scoreTPR)
    plt.show()
    print("OK")



    #loaded_model = pickle.load(open(filename, 'rb'))

    return mi, delta, clf


def rovnake_oci1():
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
    for root, dirs, files in os.walk('F://DB/eyes/output/ot/M'):
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

    for root, dirs, files in os.walk('F://DB/eyes/output/ot/binary'):
        for file in files:
            p = os.path.join(root, file)
            if p.endswith(('.bmp')):
                tmp.append(p)

    while x < 118:
        for pic in tmp:
            if(chr(x) == pic[-10]):
                tmp2.append(pic)
        oci[str('00' + str(chr(x)) + '_1')] = tmp2.copy()
        x = x + 1
        tmp2 = []

    oci2.fromkeys(oci.keys(),[])
    tmp = []
    for x,a in oci.items():
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


    for kluc, hodnota in oci2.items():
        for i in range(1,len(hodnota)):
            posun = hamming_dist_rotacia(hodnota[0], hodnota[i])
            for j in range(0, posun):
                hodnota[i] = np.roll(hodnota[i],1)
            oci2[kluc][i] = hodnota[i]


    listOci = []
    labels = []
    for kluc in oci2.keys():
        for oko in oci2[kluc]:
            labels.append(kluc)
            listOci.append(oko)

    c = list(zip(listOci, labels))
    import random
    random.shuffle(c)

    listOci, labels = zip(*c)

    from sklearn.neural_network import MLPClassifier


    trainList = []
    trainLab = []
    testList = []
    testLab = []

    for i in range(0,60):
        trainList.append(listOci[i])
        trainLab.append(labels[i])
    for i in range(50,len(masky)):
        testList.append(listOci[i])
        testLab.append(labels[i])

    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(100,100,),
                        learning_rate='adaptive', learning_rate_init=0.01,
                        max_iter=200, momentum=0.1, n_iter_no_change=20,
                        nesterovs_momentum=True, power_t=0.5, random_state=1,
                        shuffle=True, solver='adam', tol=0.0001,
                        validation_fraction=0.1, verbose=False, warm_start=False)
    #trainList = np.reshape(trainList, (100, 28)).T
    trainList1 = []
    for k in trainList:
        trainList1.append(k.reshape(170 * 493))


    testList1 = []
    for k in testList:
        testList1.append(k.reshape(170 * 493))

    clf.fit(trainList1, trainLab)

    print("Uspesnost je " + str(clf.score(testList1,testLab)))

    vysledky_z_mlp = clf.predict_proba(trainList1)


    valTPR = []
    valFPR = []
    hranica = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
    ctr = 0
    ctr1 = 0

    mi = np.mean(vysledky_z_mlp)
    delta = np.var(vysledky_z_mlp)
    print('mi = ' + str(mi) + ', delta = ' + str(delta))
    for i in range(0, vysledky_z_mlp.shape[0]):
        for j in range(0, vysledky_z_mlp.shape[1]):
            vysledky_z_mlp[i][j] = ((vysledky_z_mlp[i][j]) - mi) / delta

    print(vysledky_z_mlp.shape)
    for value_for_TPR in vysledky_z_mlp:
        label_oka = trainLab[ctr][2]
        cislo_v_riadku = ord(label_oka) - 97
        valTPR.append(value_for_TPR[cislo_v_riadku])
        ctr = ctr + 1

    for value_for_FPR in vysledky_z_mlp:
        label_oka = trainLab[ctr1][2]
        cislo_v_riadku = ord(label_oka) - 97
        for prvok in value_for_FPR:
            if prvok == value_for_FPR[cislo_v_riadku]:
                continue
            else:
                valFPR.append(prvok)
        ctr1 = ctr1 + 1

    z_scoreTPR = []
    z_scoreFPR = []

    # mi = np.mean(valTPR)
    # delta = np.var(valTPR)
    # print('mi = ' + str(mi) + ', delta = ' + str(delta))
    # for value_for_TPR in vysledky_z_mlp:
    #     label_oka = trainLab[ctr][2]
    #     cislo_v_riadku = ord(label_oka) - 97
    #     a = (value_for_TPR[cislo_v_riadku] - mi)/(delta)
    #     z_score.append(a)

    for h in hranica:
        for v in valTPR:
            if v > h:
                ctr = ctr + 1
        tmp_Z_Score = ctr/len(valTPR)
        z_scoreTPR.append(tmp_Z_Score)
        ctr = 0

    for h in hranica:
        for v in valFPR:
            if v > h:
                ctr = ctr + 1
        tmp_Z_Score = ctr/len(valFPR)
        z_scoreFPR.append(tmp_Z_Score)
        ctr = 0

    for i in range(0,len(z_scoreTPR)):
        print('TPR = ' + str(z_scoreTPR[i]) + ', FPR = ' + str(z_scoreFPR[i]))


    plt.plot(z_scoreFPR,z_scoreTPR)
    plt.show()
    print("OK")


#import warnings
#warnings.filterwarnings("ignore")



if __name__ == "__main__":
    # print("Same eye")
    # rovnake = z3_same_eye()
    # print("Diff eye")
    # rozdielne = z3_diff_eye()
    #
    # plt.hist(rozdielne, color='blue', label='Rozdielne oci')
    # plt.hist(rovnake, color='red', label='Rovnake oci')
    # plt.show()


    # import seaborn as sns
    #
    # sns.distplot(rozdielne, kde=True)
    # sns.distplot(rovnake, kde=True)
    # #plt.axis('off')
    # plt.show()




    # im = cv2.imread("F://BIOM/z4/diff_eye/1/001_1_10.bmp")
    # im_mask = cv2.imread("F://BIOM/output/ot/M/001_1_1.bmp")
    # im2 = cv2.imread("F://BIOM/z4/diff_eye/2/001_1_20.bmp")
    # im_mask2 = cv2.imread("F://BIOM/output/ot/M/001_1_2.bmp")
    # im3 = cv2.imread("F://BIOM/z4/diff_eye/2/007_1_20.bmp")
    # im_mask3 = cv2.imread("F://BIOM/output/ot/M/007_1_2.bmp")
    #
    # print(hamming_dist(im & im_mask, im & im_mask))
    # print(hamming_dist(im2 & im_mask2, im & im_mask))
    # print(hamming_dist(im & im_mask, im3 & im_mask3))
    import sys
    import subprocess

    mi, sigma, neuronka = rovnake_oci2()
