export CUDA_VISIBLE_DEVICES=1
python main.py \
 --gan_type VAEDRGAN3D \
 --dataset Bosphorus \
 --comment noFC \
 --batch_size 16 \
 --test_sample_size 12 \
 --num_workers 4 \
 --lrD 0.00001 \
 --lrG 0.0002 \
 --epoch 1200 
# --loss_option reconL1 \
# --generate True \
 #--resume True \
