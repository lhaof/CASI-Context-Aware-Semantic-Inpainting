
require 'image'
require 'cunn'
require 'cudnn'
require 'nn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchNum = 5,
    batchSize = 1,        -- number of samples to produce
               			   -- path to the generator network
    name = 'test_paris_fcgan_wfeat50',
                           -- name of the experiment and prefix of file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = 1st GPU etc.
    nc = 3,                -- # of channels in input
    display = 0,           -- Display image: 0 = false, 1 = true
    loadSize = 128,        -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. 
                           -- -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
    fineSize = 128,        -- size of random crops
    nThreads = 1,          -- # of data loading threads to use
    manualSeed = 2017,     -- 0 means random seed
    overlapPred = 4,       -- overlapping edges of center with context
    wtl2 = 0.999,

    netG = '',
    imgnet20_mean1 = 0.475,
    imgnet20_mean2 = 0.457,
    imgnet20_mean3 = 0.408,
    paris_mean1 = 117.0/255.0,
    paris_mean2 = 104.0/255.0,
    paris_mean3 = 123.0/255.0,
    mean_value1 = 0.5,
    mean_value2 = 0.5,
    mean_value3 = 0.5,	
	DATA_ROOT = 'data_root',
	dataset = 'Paris',
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt.DATA_ROOT)

-- set seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end
torch.manualSeed(opt.manualSeed)

-- load networks
assert(opt.netG ~= '', 'provide a generator model')
netG = util.load(opt.netG, opt.gpu)
netG:evaluate()

-- initialize variables
netG_input = torch.Tensor(opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
if opt.dataset == 'Paris' then
	opt.mean_value1 = opt.paris_mean1
	opt.mean_value2 = opt.paris_mean2
	opt.mean_value3 = opt.paris_mean3
end

-- criterion used to compute loss
criterion = nn.BCECriterion()
criterionABS = nn.AbsCriterion()
criterionMSE = nn.MSECriterion()

-- port to GPU
if opt.gpu > 0 then
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    if pcall(require, 'cudnn') then
    if opt.quiet == 0 then print('Using CUDNN !') end
        require 'cudnn'
        netG = util.cudnn(netG)
    end
    netG:cuda()
    criterion:cuda()
    criterionMSE:cuda()
    criterionABS:cuda()
    netG_input = netG_input:cuda()
else
    netG:float()
end
if opt.quiet == 0 then
    print(netG)
    print(criterionMSE)
end

local tp = function(...)
    local se; local si; local o = 0;
    for k,v in ipairs({...}) do
        if k==1 then se=v end;if k==2 then si=v end;if k==3 then o=v end;
    end
    return 1 + (se-si)/2 + o
end
local bt = function(...)
    local se; local si; local o = 0;
    for k,v in ipairs({...}) do
        if k==1 then se=v end;if k==2 then si=v end;if k==3 then o=v end;
    end
    return (se+si)/2 - o
end
local lf = function(...)
    local se; local si; local o = 0;
    for k,v in ipairs({...}) do
        if k==1 then se=v end;if k==2 then si=v end;if k==3 then o=v end;
    end
    return 1 + (se-si)/2 + o
end
local rg = function(...)
    local se; local si; local o = 0;
    for k,v in ipairs({...}) do
        if k==1 then se=v end;if k==2 then si=v end;if k==3 then o=v end;
    end
    return (se+si)/2 - o
end

local h; local w;
local ph; local pw;
local crop_center = function(T, csize)
    return T[{{},{},{tp(T:size(3), csize), bt(T:size(4), csize)},
        {lf(T:size(3), csize), rg(T:size(4), csize)}}]:clone()
end

local netG_l1loss56_tot = 0
local netG_l2loss56_tot = 0

local timer = torch.Timer()
paths.rmall(opt.name,'yes')
paths.mkdir(opt.name)

local filenames = {}
for f in paths.iterfiles(opt.DATA_ROOT) do
    local fullname = paths.concat(opt.DATA_ROOT, f)
	filenames[#filenames+1] = fullname
	print(fullname)
end
table.sort(filenames)

for i = 1, #filenames do
	local image_path = filenames[i]
    local image_ctx = image.load(image_path)
	image_ctx = image.scale(image_ctx, opt.loadSize)
    local netG_ctx = image_ctx:clone()
    local ground_truth = image_ctx:clone()
    -- [0, 1]

    -- save ground truth
    local gt = {}
    h = ground_truth:size(2); w = ground_truth:size(3);
    for k,v in ipairs({56}) do
        gt[v] = ground_truth[{{},{tp(h,v),bt(h,v)},{lf(w,v),rg(w,v)}}]:clone()
    end

    -- do prediction
    -- fill with imagenet20(fcgan)'s mean value
	netG_ctx:mul(2.0):add(-1.0)
    h = netG_ctx:size(2); w = netG_ctx:size(3);
    netG_ctx[{{1},{tp(h,56),bt(h,56)},{lf(w,56),rg(w,56)}}]=opt.mean_value1 * 2.0 - 1.0
    netG_ctx[{{2},{tp(h,56),bt(h,56)},{lf(w,56),rg(w,56)}}]=opt.mean_value2 * 2.0 - 1.0
    netG_ctx[{{3},{tp(h,56),bt(h,56)},{lf(w,56),rg(w,56)}}]=opt.mean_value3 * 2.0 - 1.0
    netG_input:copy(netG_ctx)
    local netG_pred = netG:forward(netG_input):clone()
    netG_pred = netG_pred[{{},{},{1 + opt.fineSize/4, opt.fineSize*3/4},{1 + opt.fineSize/4, opt.fineSize*3/4}}]:clone()
    -- 128 x 128 -> 64 x 64

    -- calculate loss
    local netG_l1loss56
    local netG_l2loss56
    netG_l1loss56 = criterionABS:forward(crop_center(netG_pred,56),gt[56]:cuda())
    netG_l2loss56 = criterionMSE:forward(crop_center(netG_pred,56),gt[56]:cuda())

    netG_l1loss56_tot = netG_l1loss56_tot + netG_l1loss56
    netG_l2loss56_tot = netG_l2loss56_tot + netG_l2loss56

    -- paste predicted center in the context
    h = netG_ctx:size(2); w = netG_ctx:size(3)
    ph = netG_pred:size(3); pw = netG_pred:size(4)
    netG_ctx[{{},{tp(h,56),bt(h,56)},{lf(w,56),rg(w,56)}}]:copy(
        netG_pred[{{},{},{tp(ph,56),bt(ph,56)},{lf(pw,56),rg(pw,56)}}])

    -- re-transform scale back to normal
    netG_ctx:add(1):mul(0.5)
    netG_input:add(1):mul(0.5)
    netG_pred:add(1):mul(0.5)

    -- save outputs in a pretty manner
    gt[56]=nil;
    netG_pred=nil;

    image.save(opt.name..'/'..opt.name..'_'..i..'.png',netG_ctx)
end
local nsamp = #filenames
print('\tnetG_l1loss56_mean:'..netG_l1loss56_tot/nsamp)
print('\tnetG_l2loss56_mean:'..netG_l2loss56_tot/nsamp)
