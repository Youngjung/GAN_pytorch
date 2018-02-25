export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type VAEGAN3D \
 --dataset Bosphorus \
 --comment lrG2e-4 \
 --batch_size 16 \
 --test_sample_size 32 \
 --lrG 0.0002 \
 --lrD 0.00001 \
 --num_workers 4 \
 --epoch 1200
# --resume True
