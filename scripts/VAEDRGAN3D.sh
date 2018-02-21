export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type VAEDRGAN3D \
 --dataset Bosphorus \
 --batch_size 12 \
 --test_sample_size 12 \
 --num_workers 4 \
 --lrD 0.00001 \
 --lrG 0.0002 \
 --epoch 1200 
# --loss_option reconL1 \
# --resume True \
# --generate True \
