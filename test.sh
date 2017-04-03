gpu_id=2
data_root=~/hfli/parisStreetView/val/paris_eval_gt
model=checkpoints/paris_fcgan_wfeat50_500_net_G.t7
test_script=test_fcgan.lua
CUDA_VISIBLE_DEVICES=${gpu_id} DATA_ROOT=${data_root} netG=${model} th ${test_script}
