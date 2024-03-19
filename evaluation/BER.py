import numpy as np
import cv2 as cv
import os
import glob


def cal_acc(prediction, label, thr=45):
    prediction = (prediction > thr)
    label = (label > thr)
    prediction_tmp = prediction.astype(np.float64)
    label_tmp = label.astype(np.float64)
    TP = np.sum(prediction_tmp * label_tmp)
    TN = np.sum((1 - prediction_tmp) * (1 - label_tmp))
    Np = np.sum(label_tmp)
    Nn = np.sum((1 - label_tmp))
    Union = np.sum(prediction_tmp) + Np - TP

    return TP, TN, Np, Nn, Union


if __name__ == '__main__':
    pre_paths = sorted(glob.glob(r"C:\Users\Wu\Desktop\weakly\results\SBU\*.png"))
    gt_paths = sorted(glob.glob(r"F:\Datasets\SBU\test\Bmask\*.png"))
    s = "\t"
    xls_file_head = "image_name" + s + "ber"
    with open("ber_statics.xls", "w") as f:
        f.write(xls_file_head + "\n")
    TP = 0
    TN = 0
    Np = 0
    Nn = 0
    for idx, pre_path in enumerate(pre_paths):
        file_name = pre_path.split("\\")[-1]
        gt_path = gt_paths[idx]
        target = cv.imread(gt_path, -1)
        prediction = cv.imread(pre_path, -1)
        TP_single, TN_single, Np_single, Nn_single, Union = cal_acc(prediction, target)
        TP += TP_single
        TN += TN_single
        Np += Np_single
        Nn += Nn_single
        ber_single = 0.5 * (2 - TP_single / Np_single - TN_single / Nn_single) * 100
        print("---------%s-th image processing----------" % (idx+1))
        with open("ber_statics.xls", "a") as f:
            content = file_name + s + str(round(ber_single, 2))
            f.write(content + "\n")

    ber_shadow = (1 - TP / Np) * 100
    ber_nonshadow = (1 - TN / Nn) * 100
    ber = 0.5 * (2 - TP / Np - TN / Nn) * 100
    print("Ber is {}, Shadow_Ber is {}, Nonshadow_ber is {}".format(round(ber, 2), round(ber_shadow, 2),
                                                                    round(ber_nonshadow, 2)))
