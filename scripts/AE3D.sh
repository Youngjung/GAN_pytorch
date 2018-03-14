export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type AE3D \
 --dataset Bosphorus \
 --batch_size 16 \
 --num_workers 8 \
 --lrG 0.0025 \
 --epoch 100
