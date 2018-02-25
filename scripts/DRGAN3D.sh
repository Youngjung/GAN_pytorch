export CUDA_VISIBLE_DEVICES=1
python main.py \
 --gan_type DRGAN3D \
 --dataset Bosphorus \
 --comment noG4 \
 --batch_size 20 \
 --test_sample_size 12 \
 --num_workers 4 \
 --lrD 0.00001 \
 --lrG 0.0002 \
 --resume True \
 --epoch 300 
# --loss_option reconL1 \
# --generate True \

