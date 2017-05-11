gpu_id=1
#data_root=../parisStreetView/train 
data_root=../download/imgnet20/train 
#train_script=train_fcgan_res_unet.lua
train_script=train_fcgan_res.lua
CUDA_VISIBLE_DEVICES=${gpu_id} DATA_ROOT=${data_root} th ${train_script}
