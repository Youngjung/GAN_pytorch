export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type DRcycleGAN3D \
 --dataset Bosphorus \
 --comment lrG_2e-4 \
 --batch_size 8 \
 --test_sample_size 12 \
 --num_workers 4 \
 --lrD 0.00001 \
 --lrG 0.0002 \
 --epoch 300
# --generate True \
# --resume True

