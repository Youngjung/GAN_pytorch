export CUDA_VISIBLE_DEVICES=1
python main.py \
 --gan_type DRGAN3D \
 --dataset Bosphorus \
 --comment centered_wReconLoss\
 --batch_size 16 \
 --test_sample_size 12 \
 --num_workers 8 \
 --lrD 0.00001 \
 --lrG 0.0025 \
 --epoch 300 
# --resume True
# --generate True \
# --use_GP True \

#python utils3D/npy2ply.py --fname results/Bosphorus/DRGAN3D_randomPcode/sampleGT.npy --dir_dest results/Bosphorus/DRGAN3D_randomPcode/
#python utils3D/npy2ply.py --fname results/Bosphorus/DRGAN3D_randomPcode/DRGAN3D_randomPcode_epoch100.npy --dir_dest results/Bosphorus/DRGAN3D_randomPcode/
