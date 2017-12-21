export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type GAN3D \
 --dataset ShapeNet \
 --batch_size 32 \
 --test_sample_size 16 \
 --lrG 0.0025 \
 --lrD 0.00001 \
 --epoch 100 \
 --use_GP True
# --dataset miniPie
