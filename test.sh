gpu_id=2
data_root=~/hfli/parisStreetView/val
test_script=test_fcgan.lua
CUDA_VISIBLE_DEVICES=${gpu_id} DATA_ROOT=${data_root} th ${test_script}
