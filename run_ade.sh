#DATA_ROOT=../data/COCO2017
DATA_ROOT=../data/ADEChallengeData2016
DATASET=ade
TASK=100-5 # [15-1, 10-1, 19-1, 15-5, 5-3, 5-1, 2-1, 2-2] add split to task 15-1
EPOCH=100 #50
BATCH=13 #32
LOSS=bce_loss
LR=0.05
THRESH=0.7
FEWSHOT=True
NUMSHOT=5
#NUMSHOT=1
#NUMSHOT=40
START=1
#START=0
MEMORY=300 #100 # [0 (for SSUL), 100 (for SSUL-M)]


#### few shot step 0
#python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --crop_val \
#               --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} \
#                --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly --pseudo \
#                --pseudo_thresh ${THRESH} --freeze --bn_freeze --unknown --w_transfer --amp \
#                --mem_size ${MEMORY} \
#                --few_shot ${FEWSHOT} --num_shot ${NUMSHOT} --start_step ${START}

##### few shot step 1 - last
python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --crop_val \
               --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} \
                --dataset ${DATASET} --task ${TASK} --lr_policy poly --pseudo \
                --pseudo_thresh ${THRESH} --freeze --bn_freeze --unknown --w_transfer --amp \
                --mem_size ${MEMORY} \
                --few_shot ${FEWSHOT} --num_shot ${NUMSHOT} --start_step ${START} \
                --ckpt ./checkpoints/deeplabv3_resnet101_ade_100-5_step_0_disjoint.pth \


##### non few shot
#python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --crop_val \
#               --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} \
#                --dataset ${DATASET} --task ${TASK} --lr_policy poly --pseudo \
#                --pseudo_thresh ${THRESH} --freeze --bn_freeze --unknown --w_transfer --amp \
#                --mem_size ${MEMORY} \
#                --ckpt ./checkpoints/deeplabv3_resnet101_ade_100-5_step_0_disjoint.pth \
#                --num_shot ${NUMSHOT} --start_step ${START}