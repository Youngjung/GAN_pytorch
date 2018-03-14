export CUDA_VISIBLE_DEVICES=1
python main.py \
 --gan_type DRGAN \
 --dataset MultiPie \
 --batch_size 64 \
 --comment singleSample \
 --num_workers 7
