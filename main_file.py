#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: main_file.py
# Author: Rahul Mishra <rmishra4@ncsu.edu>

import sys
import argparse
import numpy as np
import cv2
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

sys.path.append('../')
import data_loader as loader
import estimator

IMAGE_SIZE = 10

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', action='store_true',
                        help='Single Gaussian.')
    parser.add_argument('--model2', action='store_true',
                        help='Mixture of Gaussian.')
    parser.add_argument('--model3', action='store_true',
                        help='T- Distribution')
    parser.add_argument('--model4', action='store_true',
                        help='Factor Analysis')

    return parser.parse_args()

def load_data():
    train_x, train_y, test_x, test_y = loader.load_data_wrapper(IMAGE_SIZE)
#    print('Number of training: {}'.format(len(train_x)))
#    print('Number of testing: {}'.format(len(test_x)))
    return train_x, train_y, test_x, test_y

def save_image(model,mean_face, covar_face, mean_nonface, covar_nonface):
    cv2.imwrite("parametric_faces/{}_mean_face.jpg".format(model),mean_face)
    cv2.imwrite("parametric_faces/{}mean_nonface.jpg".format(model),mean_nonface)
    cv2.imwrite("parametric_faces/{}covar_face.jpg".format(model),np.sqrt(np.diag(covar_face).reshape((IMAGE_SIZE,IMAGE_SIZE))))
    cv2.imwrite("parametric_faces/{}covar_nonface.jpg".format(model),np.sqrt(np.diag(covar_face).reshape((IMAGE_SIZE,IMAGE_SIZE))))

    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()

def model1():
    print('Single Gaussian Model:')
    train_x, train_y, test_x, test_y = load_data()
    mean_face, covar_face, = estimator.MLE(train_x[0:1000])
    mean_nonface, covar_nonface, = estimator.MLE(train_x[1000:2000])

    save_image('m1', mean_face, covar_face, mean_nonface, covar_nonface)

    p1 = multivariate_normal.logpdf(test_x, mean_face.flatten(), covar_face)
    p2 = multivariate_normal.logpdf(test_x, mean_nonface.flatten(), covar_nonface)
#    p1[p1==0] = 0.5
#    p2[p2==0] = 0.5
    test(p1, p2, 'Single Gaussian')
    return 0

def model2():
    print('Mixture of Gaussian Model:')
    k = 5
    train_x, train_y, test_x, test_y = load_data()
    gmm_face = estimator.GMM(train_x[0:1000], k)
    gmm_nonface = estimator.GMM(train_x[1000:2000], k)
#    print("Training Face Model:")
    mean_face, covar_face, lambda_val_face = gmm_face.fit()
#    print("Training Non-Face Model:")
    mean_nonface, covar_nonface, lambda_val_nonface = gmm_nonface.fit()

    for i in range(k):
        save_image('m2_{}'.format(i), mean_face[i], covar_face[i], mean_nonface[i], covar_nonface[i])

    test_class = estimator.GMM(test_x, k)
    p1, p2 = test_class.test(mean_face, covar_face, lambda_val_face, mean_nonface, covar_nonface, lambda_val_nonface)
    print(lambda_val_face, lambda_val_nonface)
    test(p1, p2, 'Mixture Gaussian')

    return 0

def model3():
    print('t- Distribution Model:')
    v = 1000
    train_x, train_y, test_x, test_y = load_data()
    t_face = estimator.t_Distribution(train_x[0:1000], v)
    t_nonface = estimator.t_Distribution(train_x[1000:2000], v)
#    print("Training Face Model:")
    mean_face, covar_face = t_face.fit()
#    print("Training Non-Face Model:")
    mean_nonface, covar_nonface = t_nonface.fit()

    save_image('m3', mean_face, covar_face, mean_nonface, covar_nonface)

    test_class = estimator.t_Distribution(test_x, v)
    p1, p2 = test_class.test(mean_face, covar_face, mean_nonface, covar_nonface)
    a = p1>p2
    test(p1, p2, 't-Distribution')

    return 0

def model4():
    print('Factor Analysis Model:')
    k = 2
    train_x, train_y, test_x, test_y = load_data()
    fa_face = estimator.Factor_Analysis(train_x[0:1000], k)
    fa_nonface = estimator.Factor_Analysis(train_x[1000:2000], k)
#    print("Training Face Model:")
    mean_face, covar_face, phi_face = fa_face.fit()
#    print("Training Non-Face Model:")
    mean_nonface, covar_nonface, phi_nonface = fa_nonface.fit()
    

    save_image('m4', mean_face, abs(covar_face), mean_nonface, abs(covar_nonface))
    test_class = estimator.Factor_Analysis(test_x, k)
    p1, p2 = test_class.test(mean_face, covar_face, phi_face, mean_nonface, covar_nonface, phi_nonface)
    test(p1, p2, 'Factor Analysis')

    return 0

def test(p1, p2,model):
    
    TP, FN, TN, FP = [], [], [], []
    mn = np.min(p1-p2)
    mx = np.max(p1-p2)
    th_list = np.linspace(mn,mx,100)
#    print(mn,mx)
    
    for th in th_list:
        TP.append(p1[:100] - p2[:100] >= th)
        FN.append(p1[:100] - p2[:100] <  th)
        TN.append(p1[100:] - p2[100:] <  th)
        FP.append(p1[100:] - p2[100:] >= th)

    TP, FN, TN, FP = np.array(TP), np.array(FN), np.array(TN), np.array(FP)
    TPR = np.sum(TP, axis = 1)/100
    FPR = np.sum(FP, axis = 1)/100
    plt.plot(FPR,TPR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve {} model".format(model))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig("ROC Curve {} model.jpg".format(model))
    plt.show()
    
    FPR_0_5 = np.sum(p1[100:] > p2[100:])/100
    FNR_0_5 = np.sum(p1[:100] < p2[:100])/100
    MCR_0_5 = (np.sum(p1[100:] > p2[100:]) + np.sum(p1[:100] < p2[:100]))/200
    
    print('False Positive Rate', FPR_0_5)
    print('False Negative Rate', FNR_0_5)
    print('Misclassification Rate', MCR_0_5)
    

    return 0



if __name__ == '__main__':
    model1()
    model2()
    model3()
    model4()
    FLAGS = get_args()
    if FLAGS.model1:
        model1()
    if FLAGS.model2:
        model2()
    if FLAGS.model3:
        model3()
    if FLAGS.model4:
        model4()
