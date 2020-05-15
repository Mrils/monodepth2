# Our standard mono model
conda activate monodepth

# CUDA_VISIBLE_DEVICES=12 nohup python ../train.py --model_name mono_model \
# --log_dir /data/yhdata/yangming/log/mono_bt_12_encode_se_1234 --num_workers 16 --backbone resnet_SE --using_seg --se_layers 1 2 3 4 &

# CUDA_VISIBLE_DEVICES=13 nohup python ../train.py --model_name mono_model \
# --log_dir /data/yhdata/yangming/log/mono_bt_12_encode_se_234 --num_workers 16 --backbone resnet_SE --using_seg --se_layers 2 3 4 &

# CUDA_VISIBLE_DEVICES=14 nohup python ../train.py --model_name mono_model \
# --log_dir /data/yhdata/yangming/log/mono_bt_12_encode_se_34 --num_workers 16 --backbone resnet_SE --using_seg --se_layers 3 4 &

# CUDA_VISIBLE_DEVICES=15 nohup python ../train.py --model_name mono_model \
# --log_dir /data/yhdata/yangming//log/mono_bt_12_encode_se_4 --num_workers 16 --backbone resnet_SE --using_seg --se_layers 4 &

# CUDA_VISIBLE_DEVICES=12,13 nohup python ../train.py --model_name mono_model \
# --log_dir /data/yhdata/yangming/log/mono50_bt_12_encode_se_1234 --num_workers 16 --backbone resnet_SE --using_seg --se_layers 1 2 3 4 --num_layers 50 &

# CUDA_VISIBLE_DEVICES=14,15 nohup python ../train.py --model_name mono_model \
# --log_dir /data/yhdata/yangming/log/mono50_bt_12_encode_se_4 --num_workers 16 --backbone resnet_SE --using_seg --se_layers 2 3 4 --num_layers 50 &

CUDA_VISIBLE_DEVICES=12,13 nohup python ../train.py --model_name mono_model \
--log_dir /data/yhdata/yangming/log/mono50_bt_12_encode_se_1234_inputs --num_workers 16 --backbone resnet_SE --using_seg --se_layers 1 2 3 4 --num_layers 50 --using_inputs &

CUDA_VISIBLE_DEVICES=14,15 nohup python ../train.py --model_name mono_model \
--log_dir /data/yhdata/yangming/log/mono50_bt_12_encode_se_4_inputs --num_workers 16 --backbone resnet_SE --using_seg --se_layers 4 --num_layers 50 --using_inputs &