# """
# SSUL
# Copyright (c) 2021-present NAVER Corp.
# MIT License
# """
# import numpy as np
# import random
# import torch
#
# tasks_voc = {
#     "offline": {
#         0: list(range(21)),
#     },
#     "19-1": {
#         0: list(range(20)),
#         1: [20],
#     },
#     "15-5": {
#         0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
#         1: [16, 17, 18, 19, 20]
#     },
#     "15-1":
#         {
#             0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
#             1: [16],
#             2: [17],
#             3: [18],
#             4: [19],
#             5: [20]
#         },
#     "10-1":
#         {
#             0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#             1: [11],
#             2: [12],
#             3: [13],
#             4: [14],
#             5: [15],
#             6: [16],
#             7: [17],
#             8: [18],
#             9: [19],
#             10: [20]
#         },
#     "5-5": {
#         0: [0, 1, 2, 3, 4, 5],
#         1: [6, 7, 8, 9, 10],
#         2: [11, 12, 13, 14, 15],
#         3: [16, 17, 18, 19, 20]
#     },
#     "5-3": {
#         0: [0, 1, 2, 3, 4, 5],
#         1: [6, 7, 8],
#         2: [9, 10, 11],
#         3: [12, 13, 14],
#         4: [15, 16, 17],
#         5: [18, 19, 20],
#     },
#     "5-1": {
#         0 : [0, 1, 2, 3, 4, 5],
#         1 : [6, ],
#         2 : [7, ],
#         3 : [8, ],
#         4 : [9, ],
#         5 : [10, ],
#         6 : [11, ],
#         7 : [12, ],
#         8 : [13, ],
#         9 : [14, ],
#         10: [15, ],
#         11: [16, ],
#         12: [17, ],
#         13: [18, ],
#         14: [19, ],
#         15: [20, ],
#     },
#     "2-2": {
#         0 : [0, 1, 2],
#         1 : [3, 4],
#         2 : [5, 6],
#         3 : [7, 8],
#         4 : [9, 10],
#         5 : [11, 12],
#         6 : [13, 14],
#         7 : [15, 16],
#         8 : [17, 18],
#         9 : [19, 20],
#     },
#     "2-1":{
#         0 : [0, 1, 2],
#         1 : [3, ],
#         2 : [4, ],
#         3 : [5, ],
#         4 : [6, ],
#         5 : [7, ],
#         6 : [8, ],
#         7 : [9, ],
#         8 : [10, ],
#         9 : [11, ],
#         10: [12, ],
#         11: [13, ],
#         12: [14, ],
#         13: [15, ],
#         14: [16, ],
#         15: [17, ],
#         16: [18, ],
#         17: [19, ],
#         18: [20, ],
#     },
#     "15-1_b":{
#         0: [0, 12, 9, 20, 7, 15, 8, 14, 16, 5, 19, 4, 1, 13, 2, 11],
#         1: [17], 2: [3], 3: [6], 4: [18], 5: [10]
#     },
#     "15-1_c":{
#         0: [0, 13, 19, 15, 17, 9, 8, 5, 20, 4, 3, 10, 11, 18, 16, 7],
#         1: [12], 2: [14], 3: [6], 4: [1], 5: [2]
#     },
#     "15-1_d":{
#         0: [0, 15, 3, 2, 12, 14, 18, 20, 16, 11, 1, 19, 8, 10, 7, 17],
#         1: [6], 2: [5], 3: [13], 4: [9], 5: [4]
#     },
#     "15-1_e":{
#         0: [0, 7, 5, 3, 9, 13, 12, 14, 19, 10, 2, 1, 4, 16, 8, 17],
#         1: [15], 2: [18], 3: [6], 4: [11], 5: [20]
#     },
#     "15-1_f":{
#         0: [0, 7, 13, 5, 11, 9, 2, 15, 12, 14, 3, 20, 1, 16, 4, 18],
#         1: [8], 2: [6], 3: [10], 4: [19], 5: [17]
#     },
#     "15-1_g":{
#         0: [0, 7, 5, 9, 1, 15, 18, 14, 3, 20, 10, 4, 19, 11, 17, 16],
#         1: [12], 2: [8], 3: [6], 4: [2], 5: [13]
#     },
#     "15-1_h":{
#         0: [0, 12, 9, 19, 6, 4, 10, 5, 18, 14, 15, 16, 3, 8, 7, 11],
#         1: [13], 2: [2], 3: [20], 4: [17], 5: [1]
#     },
#     "15-1_i":{
#         0: [0, 13, 10, 15, 8, 7, 19, 4, 3, 16, 12, 14, 11, 5, 20, 6],
#         1: [2], 2: [18], 3: [9], 4: [17], 5: [1]
#     },
#     "15-1_j":{
#         0: [0, 1, 14, 9, 5, 2, 15, 8, 20, 6, 16, 18, 7, 11, 10, 19],
#         1: [3], 2: [4], 3: [17], 4: [12], 5: [13]
#     },
#     "15-1_k":{
#         0: [0, 16, 13, 1, 11, 12, 18, 6, 14, 5, 3, 7, 9, 20, 19, 15],
#         1: [4], 2: [2], 3: [10], 4: [8], 5: [17]
#     },
#     "15-1_l":{
#         0: [0, 10, 7, 6, 19, 16, 8, 17, 1, 14, 4, 9, 3, 15, 11, 12],
#         1: [2], 2: [18], 3: [20], 4: [13], 5: [5]
#     },
#     "15-1_m":{
#         0: [0, 18, 4, 14, 17, 12, 10, 7, 3, 9, 1, 8, 15, 6, 13, 2],
#         1: [5], 2: [11], 3: [20], 4: [16], 5: [19]
#     },
#     "15-1_n":{
#         0: [0, 5, 4, 13, 18, 14, 10, 19, 15, 7, 9, 3, 2, 8, 16, 20],
#         1: [1], 2: [12], 3: [11], 4: [6], 5: [17]
#     },
#     "15-1_o":{
#         0: [0, 9, 12, 13, 18, 7, 1, 15, 17, 10, 8, 4, 5, 20, 16, 6],
#         1: [14], 2: [19], 3: [11], 4: [2], 5: [3]
#     },
#     "15-1_p":{
#         0: [0, 9, 12, 13, 18, 2, 11, 15, 17, 10, 8, 4, 5, 20, 16, 6],
#         1: [14], 2: [19], 3: [1], 4: [7], 5: [3]
#     },
#     "15-1_q":{
#         0: [0, 3, 14, 13, 18, 2, 11, 15, 17, 10, 8, 4, 5, 20, 16, 6],
#         1: [12], 2: [19], 3: [1], 4: [7], 5: [9]
#     },
#     "15-1_r":{
#         0: [0, 3, 14, 13, 1, 2, 11, 15, 17, 7, 8, 4, 5, 9, 16, 19],
#         1: [12], 2: [6], 3: [18], 4: [10], 5: [20]
#     },
#     "15-1_s":{
#         0: [0, 3, 14, 6, 1, 2, 11, 12, 17, 7, 20, 4, 5, 9, 16, 19],
#         1: [15], 2: [13], 3: [18], 4: [10], 5: [8]
#     },
#     "15-1_t":{
#         0: [0, 3, 15, 13, 1, 2, 11, 18, 17, 7, 20, 8, 5, 9, 16, 19],
#         1: [14], 2: [6], 3: [12], 4: [10], 5: [4]
#     },
#     "15-1_u":{
#         0: [0, 3, 15, 13, 14, 6, 11, 18, 17, 7, 20, 8, 4, 9, 16, 10],
#         1: [1], 2: [2], 3: [12], 4: [19], 5: [5]
#     },
#     "15-1_v":{
#         0: [0, 1, 2, 12, 14, 6, 19, 18, 17, 5, 20, 8, 4, 9, 16, 10],
#         1: [3], 2: [15], 3: [13], 4: [11], 5: [7]
#     },
#     "15-1_w":{
#         0: [0, 1, 2, 12, 14, 13, 19, 18, 7, 11, 20, 8, 4, 9, 16, 10],
#         1: [3], 2: [15], 3: [6], 4: [5], 5: [17]
#     },
# }
#
# tasks_ade = {
#     "offline": {
#         0: [x for x in range(151)]
#     },
#     "100-50": {
#         0: [x for x in range(0, 101)],
#         1: [x for x in range(101, 151)]
#     },
#     "100-10":
#         {
#             0: [x for x in range(0, 101)],
#             1: [x for x in range(101, 111)],
#             2: [x for x in range(111, 121)],
#             3: [x for x in range(121, 131)],
#             4: [x for x in range(131, 141)],
#             5: [x for x in range(141, 151)]
#         },
#     "100-5":
#         {
#             0: [x for x in range(0, 101)],
#             1: [x for x in range(101, 106)],
#             2: [x for x in range(106, 111)],
#             3: [x for x in range(111, 116)],
#             4: [x for x in range(116, 121)],
#             5: [x for x in range(121, 126)],
#             6: [x for x in range(126, 131)],
#             7: [x for x in range(131, 136)],
#             8: [x for x in range(136, 141)],
#             9: [x for x in range(141, 146)],
#             10: [x for x in range(146, 151)]
#         },
#     "50-50":
#         {
#             0: [x for x in range(0, 51)],
#             1: [x for x in range(51, 101)],
#             2: [x for x in range(101, 151)]
#         },
#     "50-20":
#         {
#             0: [x for x in range(0, 51)],
#             1: [x for x in range(51, 71)],
#             2: [x for x in range(71, 91)],
#             3: [x for x in range(91, 111)],
#             4: [x for x in range(111, 131)],
#             5: [x for x in range(131, 151)],
#         },
# }
#
#
# def get_tasks(dataset, task, step=None):
#
#     if dataset == 'voc':
#         tasks = tasks_voc
#     elif dataset == 'ade':
#         tasks = tasks_ade
#     else:
#         NotImplementedError
#
#     if step is None:
#         return tasks[task].copy()
#
#     return tasks[task][step].copy()
#
#
# def get_dataset_list(dataset, task, step, mode, overlap=True, few_shot=False, num_shot=5):
#
#     all_dataset = open(f"datasets/data/{dataset}/{mode}_cls.txt", "r").read().splitlines()
#     #
#     # target_cls = get_tasks(dataset, task, step)
#     #
#     # if 0 in target_cls:
#     #     target_cls.remove(0)
#     #
#     # dataset_list = []
#     #
#     # if overlap:
#     #     fil = lambda c: any(x in target_cls for x in classes)
#     # else:
#     #     ######################################################################################3
#     #     target_cls_old = list(range(1, target_cls[0]))
#     #     # target_cls_old = list(range(1, 16))
#     #     #######################################################################################
#     #     target_cls_cum = target_cls + target_cls_old + [0, 255]
#     #
#     #     fil = lambda c: any(x in target_cls for x in classes) and all(x in target_cls_cum for x in c)
#     #
#     # for idx, classes in enumerate(all_dataset):
#     #     str_split = classes.split(" ")
#     #
#     #     img_name = str_split[0]
#     #     classes = [int(s)+1 for s in str_split[1:]]
#     #
#     #     if fil(classes):
#     #         dataset_list.append(img_name)
#     #
#
#     target_cls = get_tasks(dataset, task, step)
#
#     if 0 in target_cls:
#         target_cls.remove(0)
#
#     dataset_list = []
#
#     if overlap:
#         fil = lambda c: any(x in target_cls for x in classes)
#     else:
#         #######################################################################################
#         # target_cls_old = list(range(1, target_cls[0]))
#         target_cls_old = list(range(1, 16))
#         #######################################################################################
#         target_cls_cum = target_cls + target_cls_old + [0, 255]
#
#         fil = lambda c: any(x in target_cls for x in classes) and all(
#             x in target_cls_cum for x in c)
#
#     for idx, classes in enumerate(all_dataset):
#         str_split = classes.split(" ")
#
#         img_name = str_split[0]
#         classes = [int(s) + 1 for s in str_split[1:]]
#
#         if fil(classes):
#             dataset_list.append(img_name)
#
#     final_data_list = []
#     #####################################################################
#     if few_shot and mode == "train" and step > 0:
#         np.random.seed(1234)
#         seed_list = np.random.randint(0, 99999, size=(1,))
#         np.random.seed(seed_list[0])
#         random.seed(seed_list[0])
#         torch.manual_seed(seed_list[0])
#         for _ in range(num_shot):
#             idx = random.choice(dataset_list)
#             while True:
#                 if idx not in final_data_list:
#                     final_data_list.append(idx)
#                     break
#                 else:
#                     idx = random.choice(dataset_list)
#     else:
#         final_data_list = dataset_list
#
#     return final_data_list



"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""
import numpy as np
import random
import torch

task_coco = {
    "15-1-split0": {
        0: [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31,
            32, 34, 35, 36, 38, 39, 40, 42, 43, 44, 46, 47, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60,
            62, 63, 64, 66, 67, 68, 70, 71, 72, 74, 75, 76, 78, 79, 80],
        1: [1, 5, 9, 13, 17],
        2: [21, 25, 29, 33, 37],
        3: [41, 45, 49, 53, 57],
        4: [61, 65, 69, 73, 77]
    },

    "15-1-split1": {
        0: [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27, 28, 29, 31,
            32, 33, 35, 36, 37, 39, 40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 53, 55, 56, 57, 59, 60,
            61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76, 77, 79, 80],
        1: [2, 6, 10, 14, 18],
        2: [22, 26, 30, 34, 38],
        3: [42, 46, 50, 54, 58],
        4: [62, 66, 70, 74, 78]
    },
    "15-1-split2": {
        0: [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30,
            32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60,
            61, 62, 64, 65, 66, 68, 69, 70, 72, 73, 74, 76, 77, 78, 80],
        1: [3, 7, 11, 15, 19],
        2: [23, 27, 31, 35, 39],
        3: [43, 47, 51, 55, 59],
        4: [63, 67, 71, 75, 79]
    },
    "15-1-split3": {
        0: [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30,
            31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53, 54, 55, 57, 58, 59,
            61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79],
        1: [4, 8, 12, 16, 20],
        2: [24, 28, 32, 36, 40],
        3: [44, 48, 52, 56, 60],
        4: [64, 68, 72, 76, 80]
    }
}


tasks_voc = {
    "offline": {
        0: list(range(21)),
    },
    "19-1": {
        0: list(range(20)),
        1: [20],
    },
    "15-5": {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        1: [16, 17, 18, 19, 20]
    },
    ### split 0
    "15-1-split0":
        {
            0: [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            1: [1],
            2: [2],
            3: [3],
            4: [4],
            5: [5]
    },
    ### split 1
    "15-1-split1":
        {
            0: [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            1: [6],
            2: [7],
            3: [8],
            4: [9],
            5: [10]
    },
    ### split 2
    "15-1-split2":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20],
            1: [11],
            2: [12],
            3: [13],
            4: [14],
            5: [15]
    },
    ### split 3
    "15-1-split3":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [16],
            2: [17],
            3: [18],
            4: [19],
            5: [20]
    },
    "10-1":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11],
            2: [12],
            3: [13],
            4: [14],
            5: [15],
            6: [16],
            7: [17],
            8: [18],
            9: [19],
            10: [20]
        },
    "5-5": {
        0: [0, 1, 2, 3, 4, 5],
        1: [6, 7, 8, 9, 10],
        2: [11, 12, 13, 14, 15],
        3: [16, 17, 18, 19, 20]
    },
    "5-3": {
        0: [0, 1, 2, 3, 4, 5],
        1: [6, 7, 8],
        2: [9, 10, 11],
        3: [12, 13, 14],
        4: [15, 16, 17],
        5: [18, 19, 20],
    },
    "5-1": {
        0 : [0, 1, 2, 3, 4, 5],
        1 : [6, ],
        2 : [7, ],
        3 : [8, ],
        4 : [9, ],
        5 : [10, ],
        6 : [11, ],
        7 : [12, ],
        8 : [13, ],
        9 : [14, ],
        10: [15, ],
        11: [16, ],
        12: [17, ],
        13: [18, ],
        14: [19, ],
        15: [20, ],
    },
    "2-2": {
        0 : [0, 1, 2],
        1 : [3, 4],
        2 : [5, 6],
        3 : [7, 8],
        4 : [9, 10],
        5 : [11, 12],
        6 : [13, 14],
        7 : [15, 16],
        8 : [17, 18],
        9 : [19, 20],
    },
    "2-1":{
        0 : [0, 1, 2],
        1 : [3, ],
        2 : [4, ],
        3 : [5, ],
        4 : [6, ],
        5 : [7, ],
        6 : [8, ],
        7 : [9, ],
        8 : [10, ],
        9 : [11, ],
        10: [12, ],
        11: [13, ],
        12: [14, ],
        13: [15, ],
        14: [16, ],
        15: [17, ],
        16: [18, ],
        17: [19, ],
        18: [20, ],
    },
    "15-1_b":{
        0: [0, 12, 9, 20, 7, 15, 8, 14, 16, 5, 19, 4, 1, 13, 2, 11],
        1: [17], 2: [3], 3: [6], 4: [18], 5: [10]
    },
    "15-1_c":{
        0: [0, 13, 19, 15, 17, 9, 8, 5, 20, 4, 3, 10, 11, 18, 16, 7],
        1: [12], 2: [14], 3: [6], 4: [1], 5: [2]
    },
    "15-1_d":{
        0: [0, 15, 3, 2, 12, 14, 18, 20, 16, 11, 1, 19, 8, 10, 7, 17],
        1: [6], 2: [5], 3: [13], 4: [9], 5: [4]
    },
    "15-1_e":{
        0: [0, 7, 5, 3, 9, 13, 12, 14, 19, 10, 2, 1, 4, 16, 8, 17],
        1: [15], 2: [18], 3: [6], 4: [11], 5: [20]
    },
    "15-1_f":{
        0: [0, 7, 13, 5, 11, 9, 2, 15, 12, 14, 3, 20, 1, 16, 4, 18],
        1: [8], 2: [6], 3: [10], 4: [19], 5: [17]
    },
    "15-1_g":{
        0: [0, 7, 5, 9, 1, 15, 18, 14, 3, 20, 10, 4, 19, 11, 17, 16],
        1: [12], 2: [8], 3: [6], 4: [2], 5: [13]
    },
    "15-1_h":{
        0: [0, 12, 9, 19, 6, 4, 10, 5, 18, 14, 15, 16, 3, 8, 7, 11],
        1: [13], 2: [2], 3: [20], 4: [17], 5: [1]
    },
    "15-1_i":{
        0: [0, 13, 10, 15, 8, 7, 19, 4, 3, 16, 12, 14, 11, 5, 20, 6],
        1: [2], 2: [18], 3: [9], 4: [17], 5: [1]
    },
    "15-1_j":{
        0: [0, 1, 14, 9, 5, 2, 15, 8, 20, 6, 16, 18, 7, 11, 10, 19],
        1: [3], 2: [4], 3: [17], 4: [12], 5: [13]
    },
    "15-1_k":{
        0: [0, 16, 13, 1, 11, 12, 18, 6, 14, 5, 3, 7, 9, 20, 19, 15],
        1: [4], 2: [2], 3: [10], 4: [8], 5: [17]
    },
    "15-1_l":{
        0: [0, 10, 7, 6, 19, 16, 8, 17, 1, 14, 4, 9, 3, 15, 11, 12],
        1: [2], 2: [18], 3: [20], 4: [13], 5: [5]
    },
    "15-1_m":{
        0: [0, 18, 4, 14, 17, 12, 10, 7, 3, 9, 1, 8, 15, 6, 13, 2],
        1: [5], 2: [11], 3: [20], 4: [16], 5: [19]
    },
    "15-1_n":{
        0: [0, 5, 4, 13, 18, 14, 10, 19, 15, 7, 9, 3, 2, 8, 16, 20],
        1: [1], 2: [12], 3: [11], 4: [6], 5: [17]
    },
    "15-1_o":{
        0: [0, 9, 12, 13, 18, 7, 1, 15, 17, 10, 8, 4, 5, 20, 16, 6],
        1: [14], 2: [19], 3: [11], 4: [2], 5: [3]
    },
    "15-1_p":{
        0: [0, 9, 12, 13, 18, 2, 11, 15, 17, 10, 8, 4, 5, 20, 16, 6],
        1: [14], 2: [19], 3: [1], 4: [7], 5: [3]
    },
    "15-1_q":{
        0: [0, 3, 14, 13, 18, 2, 11, 15, 17, 10, 8, 4, 5, 20, 16, 6],
        1: [12], 2: [19], 3: [1], 4: [7], 5: [9]
    },
    "15-1_r":{
        0: [0, 3, 14, 13, 1, 2, 11, 15, 17, 7, 8, 4, 5, 9, 16, 19],
        1: [12], 2: [6], 3: [18], 4: [10], 5: [20]
    },
    "15-1_s":{
        0: [0, 3, 14, 6, 1, 2, 11, 12, 17, 7, 20, 4, 5, 9, 16, 19],
        1: [15], 2: [13], 3: [18], 4: [10], 5: [8]
    },
    "15-1_t":{
        0: [0, 3, 15, 13, 1, 2, 11, 18, 17, 7, 20, 8, 5, 9, 16, 19],
        1: [14], 2: [6], 3: [12], 4: [10], 5: [4]
    },
    "15-1_u":{
        0: [0, 3, 15, 13, 14, 6, 11, 18, 17, 7, 20, 8, 4, 9, 16, 10],
        1: [1], 2: [2], 3: [12], 4: [19], 5: [5]
    },
    "15-1_v":{
        0: [0, 1, 2, 12, 14, 6, 19, 18, 17, 5, 20, 8, 4, 9, 16, 10],
        1: [3], 2: [15], 3: [13], 4: [11], 5: [7]
    },
    "15-1_w":{
        0: [0, 1, 2, 12, 14, 13, 19, 18, 7, 11, 20, 8, 4, 9, 16, 10],
        1: [3], 2: [15], 3: [6], 4: [5], 5: [17]
    },
}

tasks_ade = {
    "offline": {
        0: [x for x in range(151)]
    },
    "100-50": {
        0: [x for x in range(0, 101)],
        1: [x for x in range(101, 151)]
    },
    "100-10":
        {
            0: [x for x in range(0, 101)],
            1: [x for x in range(101, 111)],
            2: [x for x in range(111, 121)],
            3: [x for x in range(121, 131)],
            4: [x for x in range(131, 141)],
            5: [x for x in range(141, 151)]
        },
    "100-5":
        {
            0: [x for x in range(0, 101)],
            1: [x for x in range(101, 106)],
            2: [x for x in range(106, 111)],
            3: [x for x in range(111, 116)],
            4: [x for x in range(116, 121)],
            5: [x for x in range(121, 126)],
            6: [x for x in range(126, 131)],
            7: [x for x in range(131, 136)],
            8: [x for x in range(136, 141)],
            9: [x for x in range(141, 146)],
            10: [x for x in range(146, 151)]
        },
    "50-50":
        {
            0: [x for x in range(0, 51)],
            1: [x for x in range(51, 101)],
            2: [x for x in range(101, 151)]
        },
    "50-20":
        {
            0: [x for x in range(0, 51)],
            1: [x for x in range(51, 71)],
            2: [x for x in range(71, 91)],
            3: [x for x in range(91, 111)],
            4: [x for x in range(111, 131)],
            5: [x for x in range(131, 151)],
        },
}


def get_tasks(dataset, task, step=None):

    if dataset == 'voc':
        tasks = tasks_voc
    elif dataset == 'ade':
        tasks = tasks_ade
    elif dataset == 'coco':
        tasks = task_coco
    else:
        NotImplementedError

    if step is None:
        return tasks[task].copy()

    return tasks[task][step].copy()


def get_dataset_list(dataset, task, step, mode, overlap=True):

    all_dataset = open(f"datasets/data/{dataset}/{mode}_cls.txt", "r").read().splitlines()
    #
    # target_cls = get_tasks(dataset, task, step)
    #
    # if 0 in target_cls:
    #     target_cls.remove(0)
    #
    # dataset_list = []
    #
    # if overlap:
    #     fil = lambda c: any(x in target_cls for x in classes)
    # else:
    #     ######################################################################################3
    #     target_cls_old = list(range(1, target_cls[0]))
    #     # target_cls_old = list(range(1, 16))
    #     #######################################################################################
    #     target_cls_cum = target_cls + target_cls_old + [0, 255]
    #
    #     fil = lambda c: any(x in target_cls for x in classes) and all(x in target_cls_cum for x in c)
    #
    # for idx, classes in enumerate(all_dataset):
    #     str_split = classes.split(" ")
    #
    #     img_name = str_split[0]
    #     classes = [int(s)+1 for s in str_split[1:]]
    #
    #     if fil(classes):
    #         dataset_list.append(img_name)
    #

    target_cls = get_tasks(dataset, task, step)

    if 0 in target_cls:
        target_cls.remove(0)

    dataset_list = []

    if overlap:
        fil = lambda c: any(x in target_cls for x in classes)
    else:
        #######################################################################################
        # target_cls_old = list(range(1, target_cls[0]))
        # split 0
        # target_cls_old = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] + list(range(1, target_cls[0]))
        # split 1
        # target_cls_old = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] + list(range(6, target_cls[0]))
        # split 2
        # target_cls_old = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20] + list(range(11, target_cls[0]))
        # split 3
        target_cls_old = list(range(1, 16))
        # target_cls_old = list(range(1, target_cls[0]))
        #######################################################################################
        target_cls_cum = target_cls + target_cls_old + [0, 255]

        fil = lambda c: any(x in target_cls for x in classes) and all(
            x in target_cls_cum for x in c)

    for idx, classes in enumerate(all_dataset):
        str_split = classes.split(" ")

        img_name = str_split[0]
        classes = [int(s) + 1 for s in str_split[1:]]

        if fil(classes):
            dataset_list.append(img_name)

    return dataset_list
