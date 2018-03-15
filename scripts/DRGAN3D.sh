export CUDA_VISIBLE_DEVICES=1
python main.py \
 --gan_type DRGAN3D \
 --dataset Bosphorus \
 --comment expr12 \
 --batch_size 20 \
 --num_workers 4 \
 --lrD 0.00001 \
 --lrG 0.0002 \
 --epoch 300 
# --resume True \
# --loss_option reconL1 \
# --generate True \

