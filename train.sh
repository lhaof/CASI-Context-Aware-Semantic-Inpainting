gpu_id=0
data_root=~/hfli/parisStreetView/train 
train_script=train_fcgan6.lua
CUDA_VISIBLE_DEVICES=${gpu_id} DATA_ROOT=${data_root} th ${train_script}
