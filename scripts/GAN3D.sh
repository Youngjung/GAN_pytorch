export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type GAN3D \
 --dataset Bosphorus \
 --comment Nz50 \
 --batch_size 20 \
 --test_sample_size 16 \
 --lrG 0.0002 \
 --lrD 0.00001 \
 --num_workers 4 \
 --epoch 1200
