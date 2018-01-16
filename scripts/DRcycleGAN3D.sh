export CUDA_VISIBLE_DEVICES=0
python main.py \
 --gan_type DRcycleGAN3D \
 --dataset Bosphorus \
 --batch_size 6 \
 --test_sample_size 12 \
 --num_workers 7 \
 --lrD 0.00001 \
 --lrG 0.0025 \
 --epoch 100
# --generate True \
# --use_GP True \
# --resume True

python utils3D/npy2ply.py --fname results/Bosphorus/DRcycleGAN3D/sampleGT.npy --dir_dest results/Bosphorus/DRcycleGAN3D/
python utils3D/npy2ply.py --fname results/Bosphorus/DRcycleGAN3D/DRcycleGAN3D_epoch101_recon.npy --dir_dest results/Bosphorus/DRcycleGAN3D/
python utils3D/npy2ply.py --fname results/Bosphorus/DRcycleGAN3D/DRcycleGAN3D_epoch101.npy --dir_dest results/Bosphorus/DRcycleGAN3D/
