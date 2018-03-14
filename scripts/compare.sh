export CUDA_VISIBLE_DEVICES=1
python compare.py \
 --gan_type1 DRGAN3D \
 --gan_type2 VAEGAN3D \
 --dataset Bosphorus \
 --comment1 G4_ep300 \
 --comment2 lrG2e-4 \
 --batch_size 20 \
 --num_workers 4 

