export CUDA_VISIBLE_DEVICES=1
python main.py \
 --gan_type GAN3D \
 --dataset Bosphorus \
 --batch_size 64 \
 --test_sample_size 4 \
 --lrG 0.0025 \
 --lrD 0.00001 \
 --epoch 100
# --dataset miniPie
