# import os
# import random
# import torch.utils.data as data
# from torch import distributed
# import torchvision as tv
# import numpy as np
# from .utils import Subset, filter_images, group_images
# import torch
#
# from PIL import Image
#
# from .coco_base import COCOSeg
# from .coco_20i import COCO20iReader
# from .baseset import base_set
# import tasks
#
# cfg = {'DATASET': {
#            'TRANSFORM': {
#                 'TRAIN': {
#                     'transforms': ('normalize', ),
#                     'joint_transforms': ('joint_random_scale_crop', 'joint_random_horizontal_flip'),
#                     'TRANSFORMS_DETAILS': {
#                         'NORMALIZE': {
#                             'mean': (0.485, 0.456, 0.406),
#                             'sd': (0.229, 0.224, 0.225),
#                         },
#                         'crop_size': (512, 512)
#                     }
#                 },
#                 'TEST': {
#                     'transforms': ('normalize', ),
#                     'joint_transforms': ('joint_random_scale_crop', 'joint_random_horizontal_flip'),
#                     'TRANSFORMS_DETAILS': {
#                         'NORMALIZE': {
#                             'mean': (0.485, 0.456, 0.406),
#                             'sd': (0.229, 0.224, 0.225),
#                         },
#                         'crop_size': (512, 512)
#                     }
#                 }}}}
#
# class COCOSegmentationIncremental(data.Dataset):
#     def __init__(self,
#                  root,
#                  task,
#                  train=True,
#                  transform=None,
#                  labels=None,
#                  labels_old=None,
#                  idxs_path=None,
#                  masking=True,
#                  overlap=True,
#                  step=0,
#                  few_shot=False,
#                  num_shot=5,
#                  batch_size=24,
#                  folding=3):
#
#         COCO_PATH = os.path.join(root, "COCO2017")
#         labels, labels_old, path_base = tasks.get_task_labels('coco', name=task, step=step)
#
#         if step == 0:
#             if train:
#                 ds = COCO20iReader(COCO_PATH, folding, True, exclude_novel=True)
#                 self.dataset = base_set(ds, "train", cfg)
#             else:
#                 ds = COCO20iReader(COCO_PATH, folding, False, exclude_novel=False)
#                 self.dataset = base_set(ds, "test", cfg)
#                 # ds = COCOSeg(COCO_PATH, False)
#                 # self.dataset = base_set(ds, "test", cfg)
#         else:
#             if train:
#                 ds = COCOSeg(COCO_PATH, True)
#                 dataset = base_set(ds, "test", cfg)  # Use test config to keep original scale of the image.
#
#                 #######################################
#                 idxs = list(range(len(ds)))
#                 final_file_name = []
#                 if few_shot:
#                     seed = 2022
#                     np.random.seed(seed)
#                     random.seed(seed)
#                     torch.manual_seed(seed)
#                     for k in labels:
#                         for _ in range(num_shot):
#                             idx = random.choice(idxs)
#                             while True:
#                                 novel_img_chw, mask_hw = dataset[idx]
#                                 pixel_sum = torch.sum(mask_hw == k)
#                                 # If the selected sample is bad (more than 1px) and has not been selected,
#                                 # we choose the example.
#                                 if pixel_sum > 1 and idx not in final_file_name:
#                                     final_file_name.append(idx)
#                                     break
#                                 else:
#                                     idx = random.choice(idxs)
#                     assert len(final_file_name) == num_shot*len(labels)
#                 else:
#                     final_file_name = idxs
#
#                 idxs = final_file_name
#
#                 while len(idxs) < batch_size:
#                     if num_shot == 5:
#                         idxs = idxs * 20
#                     elif num_shot == 1:
#                         idxs = idxs * 100
#                     else:
#                         idxs = idxs * 5
#
#                 self.dataset = Subset(dataset, idxs)
#                 #######################################
#
#             else:
#                 ds = COCOSeg(COCO_PATH, False)
#                 self.dataset = base_set(ds, "test", cfg)
#
#     def __getitem__(self, index):
#         return self.dataset[index]
#
#     def __len__(self):
#         return len(self.dataset)
#
#     @staticmethod
#     def __strip_zero(labels):
#         while 0 in labels:
#             labels.remove(0)


import os
import random
import torch.utils.data as data
from torch import distributed
import torchvision as tv
import numpy as np
from .utils import Subset #, filter_images, group_images
import torch
import json

from PIL import Image

from .coco_base import COCOSeg
from .coco_20i import COCO20iReader
from .baseset import base_set
from utils import tasks

cfg = {'DATASET': {
           'TRANSFORM': {
                'TRAIN': {
                    'transforms': ('normalize', ),
                    'joint_transforms': ('joint_random_scale_crop', 'joint_random_horizontal_flip'),
                    'TRANSFORMS_DETAILS': {
                        'NORMALIZE': {
                            'mean': (0.485, 0.456, 0.406),
                            'sd': (0.229, 0.224, 0.225),
                        },
                        'crop_size': (512, 512)
                    }
                },
                'TEST': {
                    'transforms': ('normalize', ),
                    'joint_transforms': ('joint_random_scale_crop', 'joint_random_horizontal_flip'),
                    'TRANSFORMS_DETAILS': {
                        'NORMALIZE': {
                            'mean': (0.485, 0.456, 0.406),
                            'sd': (0.229, 0.224, 0.225),
                        },
                        'crop_size': (512, 512)
                    }
                }}}}

class COCOSegmentation(data.Dataset):
    def __init__(self,
                 opts,
                 image_set='train',
                 transform=None,
                 cil_step=0,
                 mem_size=0):

        self.root = opts.data_root
        self.task = opts.task
        self.overlap = opts.overlap
        self.unknown = opts.unknown

        self.image_set = image_set
        self.folding = opts.folding
        self.few_shot = opts.few_shot
        self.num_shot = opts.num_shot

        # COCO_PATH = os.path.join(self.root, "COCO2017")
        COCO_PATH = self.root
        self.target_cls = tasks.get_tasks('coco', self.task, cil_step)
        self.target_cls += [255]  # including ignore index (255)

        ########################################################################################
        if cil_step == 0:
            if image_set == 'train':
                ds = COCO20iReader(COCO_PATH, self.folding, True, exclude_novel=True)
                self.dataset = base_set(ds, "train", cfg)
            else:
                ds = COCO20iReader(COCO_PATH, self.folding, False, exclude_novel=False)
                self.dataset = base_set(ds, "test", cfg)
                # ds = COCOSeg(COCO_PATH, False)
                # self.dataset = base_set(ds, "test", cfg)
        else:
            if image_set == 'train':
                ds = COCOSeg(COCO_PATH, True)
                dataset = base_set(ds, "test", cfg)  # Use test config to keep original scale of the image.

                #######################################
                idxs = list(range(len(ds)))
                final_file_name = []
                if self.few_shot:
                    seed = 2022
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    for k in self.target_cls:
                        for _ in range(self.num_shot):
                            idx = random.choice(idxs)
                            while True:
                                novel_img_chw, mask_hw = dataset[idx]
                                pixel_sum = torch.sum(mask_hw == k)
                                # If the selected sample is bad (more than 1px) and has not been selected,
                                # we choose the example.
                                if pixel_sum > 1 and idx not in final_file_name:
                                    final_file_name.append(idx)
                                    break
                                else:
                                    idx = random.choice(idxs)
                    assert len(final_file_name) == self.num_shot*len(self.target_cls)
                else:
                    final_file_name = idxs

                idxs = final_file_name

                while len(idxs) < opts.batch_size:
                    if self.num_shot == 5:
                        idxs = idxs * 20
                    elif self.num_shot == 1:
                        idxs = idxs * 100
                    else:
                        idxs = idxs * 5

                self.dataset = Subset(dataset, idxs)
                #######################################
            elif image_set == 'memory':
                for s in range(cil_step):
                    self.target_cls += tasks.get_tasks('ade', self.task, s)

                coco_root = './datasets/data/coco'
                memory_json = os.path.join(coco_root, 'memory.json')

                with open(memory_json, "r") as json_file:
                    memory_list = json.load(json_file)

                file_names = memory_list[f"step_{cil_step}"]["memory_list"]
                print("... memory list : ", len(file_names), self.target_cls)

                while len(file_names) < opts.batch_size:
                    file_names = file_names * 2
            else:
                ds = COCOSeg(COCO_PATH, False)
                self.dataset = base_set(ds, "test", cfg)
        #
        #
        #
        #
        # if image_set == 'test':
        #     file_names = open(os.path.join(ade_root, 'val.txt'), 'r')
        #     file_names = file_names.read().splitlines()
        #
        # elif image_set == 'memory':
        #     for s in range(cil_step):
        #         self.target_cls += get_tasks('ade', self.task, s)
        #
        #     memory_json = os.path.join(ade_root, 'memory.json')
        #
        #     with open(memory_json, "r") as json_file:
        #         memory_list = json.load(json_file)
        #
        #     file_names = memory_list[f"step_{cil_step}"]["memory_list"]
        #     print("... memory list : ", len(file_names), self.target_cls)
        #
        #     while len(file_names) < opts.batch_size:
        #         file_names = file_names * 2
        #
        # else:
        #     file_names = get_dataset_list('ade', self.task, cil_step, image_set, self.overlap)
        ########################################################################################

















        # labels, labels_old, path_base = tasks.get_task_labels('coco', name=self.task, step=step)


    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)
