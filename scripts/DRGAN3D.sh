export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type DRGAN3D \
 --dataset Bosphorus \
 --comment randomPcode\
 --batch_size 16 \
 --test_sample_size 12 \
 --num_workers 7 \
 --lrD 0.00001 \
 --lrG 0.0025 \
 --epoch 100
# --generate True \
# --use_GP True \
# --resume True

python utils3D/npy2ply.py --fname results/Bosphorus/DRGAN3D_randomPcode/sampleGT.npy --dir_dest results/Bosphorus/DRGAN3D_randomPcode/
python utils3D/npy2ply.py --fname results/Bosphorus/DRGAN3D_randomPcode/DRGAN3D_randomPcode_epoch100.npy --dir_dest results/Bosphorus/DRGAN3D_randomPcode/
