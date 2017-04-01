gpu_id=1
data_root=~/hfli/parisStreetView/val
model=checkpoints/paris_fcgan_wfeat0.5_500_net_G.t7
test_script=test_fcgan.lua
CUDA_VISIBLE_DEVICES=${gpu_id} DATA_ROOT=${data_root} netG=${model} th ${test_script}
