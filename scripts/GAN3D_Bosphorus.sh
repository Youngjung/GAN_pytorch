export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type GAN3D \
 --dataset Bosphorus \
 --comment projected \
 --batch_size 24 \
 --test_sample_size 8 \
 --lrG 0.0025 \
 --lrD 0.00001 \
 --num_workers 8 \
 --epoch 100
