export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type DRGAN3D \
 --dataset Bosphorus \
 --comment recon \
 --batch_size 16 \
 --test_sample_size 12 \
 --num_workers 7 \
 --lrD 0.00001 \
 --epoch 100
# --generate True \
# --use_GP True \
# --resume True
