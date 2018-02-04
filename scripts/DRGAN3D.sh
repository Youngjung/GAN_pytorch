export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type DRGAN3D \
 --dataset Bosphorus \
 --comment lrG_2e-4 \
 --batch_size 8 \
 --test_sample_size 12 \
 --num_workers 4 \
 --lrD 0.00001 \
 --lrG 0.0002 \
 --loss_option dist \
 --epoch 300 
# --resume True \
# --centerBosphorus False \
# --generate True \
# --use_GP True \

