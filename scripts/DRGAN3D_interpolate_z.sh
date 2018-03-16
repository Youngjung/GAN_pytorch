export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type DRGAN3D \
 --dataset Bosphorus \
 --comment G4_ep300 \
 --batch_size 1 \
 --num_workers 4 \
 --lrD 0.00001 \
 --lrG 0.0002 \
 --interpolate z\
 --fname_cache cache_Bosphorus_predef.txt

