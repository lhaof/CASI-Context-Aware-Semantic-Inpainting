gpu_id=2
data_root=~/hfli/parisStreetView/train 
train_script=train_fcgan_res.lua
CUDA_VISIBLE_DEVICES=${gpu_id} DATA_ROOT=${data_root} th ${train_script}
