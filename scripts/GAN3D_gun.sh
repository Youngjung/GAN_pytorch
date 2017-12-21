export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type GAN3D \
 --dataset ShapeNet \
 --synsetId gun \
 --comment gun \
 --batch_size 64 \
 --test_sample_size 16 \
 --lrG 0.0025 \
 --lrD 0.00001 \
 --epoch 100
# --dataset miniPie
