DATA_ROOT=../data/COCO2017
DATASET=coco
TASK=15-1-split0
FOLDING=0
EPOCH=50 #50
BATCH=16 #32
LOSS=bce_loss
LR=0.001 # 0.01# try decreasing learning rate
THRESH=0.7
FEWSHOT=True
NUMSHOT=5
START=1
#START=0
MEMORY=100 # [0 (for SSUL), 100 (for SSUL-M)]

##### few shot step 0
#python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --crop_val \
#               --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} \
#                --dataset ${DATASET} --task ${TASK} --folding ${FOLDING} --lr_policy poly --pseudo \
#                --pseudo_thresh ${THRESH} --freeze --bn_freeze --unknown --w_transfer --amp \
#                --mem_size ${MEMORY} \
#                --few_shot ${FEWSHOT} --num_shot ${NUMSHOT} --start_step ${START}

##### few shot step 1 - last
python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --crop_val \
               --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} \
                --dataset ${DATASET} --task ${TASK} --folding ${FOLDING} --lr_policy poly --pseudo \
                --pseudo_thresh ${THRESH} --freeze --bn_freeze --unknown --w_transfer --amp \
                --mem_size ${MEMORY} \
                --few_shot ${FEWSHOT} --num_shot ${NUMSHOT} --start_step ${START} \
                --ckpt ./checkpoints/deeplabv3_resnet101_coco_15-1-split1_step_0_disjoint.pth \


###### non few shot
#python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --crop_val \
#               --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} \
#                --dataset ${DATASET} --task ${TASK} --folding ${FOLDING} --lr_policy poly --pseudo \
#                --pseudo_thresh ${THRESH} --freeze --bn_freeze --unknown --w_transfer --amp \
#                --mem_size ${MEMORY} \
#                --ckpt ./checkpoints/deeplabv3_resnet101_coco_15-1-split1_step_0_disjoint.pth \
#                --num_shot ${NUMSHOT} --start_step ${START}
