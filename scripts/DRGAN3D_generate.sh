export CUDA_VISIBLE_DEVICES=1
python main.py \
 --gan_type DRGAN3D \
 --dataset Bosphorus \
 --comment G4_ep300 \
 --batch_size 64 \
 --test_sample_size 12 \
 --num_workers 4 \
 --lrD 0.00001 \
 --lrG 0.0002 \
 --generate True \
 --epoch 300 

