
require 'image'
require 'cunn'
require 'cudnn'
require 'nn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = 
    batchNum = 5,
    batchSize = 64,        -- number of samples to produce
                           -- path to the generator network
    name = 'test_paris_fcgan_wfeat0.5',      
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
	mean_value1 = ,
	mean_value2 = ,
	mean_value3 = ,
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

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
input_image_ctx = torch.Tensor(opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize)

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
    input_image_ctx = input_image_ctx:cuda()
	label = label:cuda()
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
local bcnt = 0
paths.rmall(opt.name,'yes')
paths.mkdir(opt.name)

for bat = 1, opt.batchNum do

    local image_ctx = data:getBatch()
    local refNetG_ctx = image_ctx:clone()
    local netG_ctx = image_ctx:clone()
    local ground_truth = image_ctx:clone()
	-- [-1, 1]

	-- save ground truth
	local gt = {}
	h = ground_truth:size(3); w = ground_truth:size(4);
	for k,v in ipairs({56,64}) do
		gt[v] = ground_truth[{{},{},{tp(h,v),bt(h,v)},{lf(w,v),rg(w,v)}}]:clone()
	end
	bcnt = bcnt + 1

	-- do prediction
	-- fill with imagenet20(fcgan)'s mean value
	h = netG_ctx:size(3); w = netG_ctx:size(4);
	netG_ctx[{{},{1},{tp(h,56),bt(h,56)},{lf(w,56),rg(w,56)}}]=2*0.475-1.0
	netG_ctx[{{},{2},{tp(h,56),bt(h,56)},{lf(w,56),rg(w,56)}}]=2*0.457-1.0
	netG_ctx[{{},{3},{tp(h,56),bt(h,56)},{lf(w,56),rg(w,56)}}]=2*0.408-1.0
	netG_input:copy(netG_ctx)
    local netG_pred = netG:forward(netG_input):clone()
	netG_pred = netG_pred[{{},{},{1 + opt.fineSize/4, opt.fineSize*3/4},{1 + opt.fineSize/4, opt.fineSize*3/4}}]:clone()
	-- 128 x 128 -> 64 x 64

	-- calculate loss
	local netG_l1loss56
	local netG_l2loss56
	local netG_advloss
	local netG_gloss
	netG_l1loss64 = criterionABS:forward(netG_pred,gt[64]:cuda())
	netG_l2loss64 = criterionMSE:forward(netG_pred,gt[64]:cuda())
	netG_l2loss56 = criterionMSE:forward(crop_center(netG_pred,56),gt[56]:cuda())

	local output = netD:forward(netG_pred):clone()
	label:fill(1)
	netG_advloss = criterion:forward(output, label)  -- generator aims at approaching real_label
	netG_gloss = opt.w_rec * netG_l2loss64 + opt.w_adv * netG_advloss

    netG_l1loss64_tot = netG_l1loss64_tot + netG_l1loss64
	netG_l2loss64_tot = netG_l2loss64_tot + netG_l2loss64
	netG_l2loss56_tot = netG_l2loss56_tot + netG_l2loss56
	netG_gloss_tot = netG_gloss_tot + netG_gloss

	if ( (opt.display_process == 1) and (bcnt % opt.disp_period == 0) ) then
		print('bat:'..bat..' bcnt:'..bcnt
		            ..'\trefNetG_l1loss64:'..refNetG_l1loss64..' netG_l1loss64:'..netG_l1loss64
					..' refNetG_l2loss64:'..refNetG_l2loss64..' netG_l2loss64:'..netG_l2loss64
					..' refNetG_l2loss56:'..refNetG_l2loss56..' netG_l2loss56:'..netG_l2loss56
					..' refNetG_advloss:'..refNetG_advloss..' netG_advloss:'..netG_advloss
					..' refNetG_gloss:'..refNetG_gloss..' netG_gloss:'..netG_gloss)
	end

    -- paste predicted center in the context
	h = netG_ctx:size(3); w = netG_ctx:size(4)
	ph = netG_pred:size(3); pw = netG_pred:size(4)
	netG_ctx[{{},{},{tp(h,56),bt(h,56)},{lf(w,56),rg(w,56)}}]:copy(
		netG_pred[{{},{},{tp(ph,56),bt(ph,56)},{lf(pw,56),rg(pw,56)}}])

    -- re-transform scale back to normal
    input_image_ctx:add(1):mul(0.5)
    image_ctx:add(1):mul(0.5)
	netG_ctx:add(1):mul(0.5)

	netG_input:add(1):mul(0.5)

	netG_pred:add(1):mul(0.5)

	ground_truth:add(1):mul(0.5)
	gt[64]:add(1):mul(0.5)
	gt[56]:add(1):mul(0.5)

    -- calculate feature loss
	featNet_input[{{},{1},{},{}}]:fill(0.485)
	featNet_input[{{},{2},{},{}}]:fill(0.456)
	featNet_input[{{},{3},{},{}}]:fill(0.406)
	featNet_input[{{},{},{1 + (224-opt.fineSize)/2, (224+opt.fineSize)/2},{1 + (224-opt.fineSize)/2, (224+opt.fineSize)/2}}]:copy(ground_truth)

	featNet_input[{{},{1},{},{}}]:add(-0.485):div(0.229)
	featNet_input[{{},{2},{},{}}]:add(-0.456):div(0.224)
	featNet_input[{{},{3},{},{}}]:add(-0.406):div(0.225)
	local gt_feat = featNet:forward(featNet_input):clone()

	featNet_input[{{},{1},{},{}}]:fill(0.485)
	featNet_input[{{},{2},{},{}}]:fill(0.456)
	featNet_input[{{},{3},{},{}}]:fill(0.406)
	featNet_input[{{},{},{1 + (224-opt.fineSize)/2, (224+opt.fineSize)/2},{1 + (224-opt.fineSize)/2, (224+opt.fineSize)/2}}]:copy(netG_ctx)

	featNet_input[{{},{1},{},{}}]:add(-0.485):div(0.229)
	featNet_input[{{},{2},{},{}}]:add(-0.456):div(0.224)
	featNet_input[{{},{3},{},{}}]:add(-0.406):div(0.225)
	local netG_feat = featNet:forward(featNet_input):clone()

	local netG_floss = criterionMSE:forward(netG_feat, gt_feat)
	netG_floss_tot = netG_floss_tot + netG_floss
	netG_gloss_tot = netG_gloss_tot + opt.w_feat * netG_floss

    -- save outputs in a pretty manner
    gt[64]=nil; gt[56]=nil;
	refNetG_pred=nil; netG_pred=nil;
    local pretty_output = torch.Tensor(4*opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)

	h = refNetG_input:size(3); w = refNetG_input:size(4)
	refNetG_input[{{},{},{tp(h,56),bt(h,56)},{lf(w,56),rg(w,56)}}] = 1

    for i=1,opt.batchSize do
        pretty_output[4*i-3]:copy(refNetG_ctx[i])
        pretty_output[4*i-2]:copy(netG_ctx[i])
        pretty_output[4*i-1]:copy(refNetG_input[i])
		pretty_output[4*i]:copy(ground_truth[i])
    end
	if (bcnt % opt.save_period == 0) then
    	image.save(opt.name..'/'..opt.name..'_bcnt'..bcnt..'.png',image.toDisplayTensor(pretty_output))
		if (opt.display_save == 1 ) then
	    	print('Saved predictions to: ./'..opt.name..'_b'..bcnt..'.png')
		end
	end

	::continue::
end

if (opt.display_result == 1) then
	print('batchNum:'..opt.batchNum..' bcnt:'..bcnt)
	assert(bcnt ~= 0)
	print('\tnetG_l1loss56_mean:'..netG_l1loss56_tot/bcnt)
	print('\tnetG_l2loss56_mean:'..netG_l2loss56_tot/bcnt)
	print('\tnetG_gloss_mean:'..netG_gloss_tot/bcnt)
	print('\tnetG_floss_mean:'..netG_floss_tot/bcnt)
end

if (opt.display_time == 1) then
	print('Time elapsed: '..timer:time().real..' seconds')
end
