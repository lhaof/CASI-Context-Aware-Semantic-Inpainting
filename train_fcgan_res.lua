require 'image'
require 'torch'
require 'cudnn'
require 'cunn'
require 'nn'
require 'optim'
util = paths.dofile('util.lua')

opt = {
   batchSize = 64,         -- number of samples to produce
   loadSize = 160,         -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. 
                           -- -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
   fineSize = 128,         -- size of random crops
   nBottleneck = 4000,     -- #  of dim for bottleneck of encoder
   nef = 64,               -- #  of encoder filters in first conv layer
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nc = 3,                 -- # of channels in input
   wtl2 = 0.999,           -- 0 means don't use else use with this weight
   overlapPred = 4,        -- overlapping edges
   nThreads = 4,           -- #  of data loading threads to use
   niter = 1000,           -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 0,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   display_iter = 50,      -- # number of iterations after which display is updated
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'paris_fcgan_res_wfeat0',   -- name of the experiment you are running
   manualSeed = 0,         -- 0 means random seed

   -- Extra Options:
   conditionAdv = 0,       -- 0 means false else true
   noiseGen = 0,           -- 0 means false else true
   noisetype = 'normal',   -- uniform / normal
   nz = 100,               -- #  of dim for Z

   featNet = '/home/liguanbin/hfli/download/resnet-18.t7',
   loadNetG = '',
   loadNetD = '',
   begin_epoch = 1,

   w_adv = 1,
   --w_rec = 799,
   --w_feat = 200,
   --w_rec = 949,
   --w_feat = 50,
   w_rec = 999,
   w_feat = 0,

   -- resnet mean & std
   featNet_mean1 = 0.485,
   featNet_mean2 = 0.456,
   featNet_mean3 = 0.406,
   featNet_std1 = 0.229,
   featNet_std2 = 0.224,
   featNet_std3 = 0.225,

   imgnet20_mean1 = 0.475,
   imgnet20_mean2 = 0.457,
   imgnet20_mean3 = 0.408,
   paris_mean1 = 117.0/255.0,
   paris_mean2 = 104.0/255.0,
   paris_mean3 = 123.0/255.0,
   mean_value1 = 0.5,
   mean_value2 = 0.5,
   mean_value3 = 0.5,

   dataset = 'Paris',
   --dataset = 'ImageNet',
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

-- set seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())

-- initialize loss weights
local weight_sum = opt.w_adv + opt.w_feat + opt.w_rec
opt.w_adv = opt.w_adv * 1.0 / weight_sum
opt.w_feat = opt.w_feat * 1.0 / weight_sum
opt.w_rec = opt.w_rec * 1.0 / weight_sum
print("w_adv:"..opt.w_adv.." w_feat:"..opt.w_feat.." w_rec:"..opt.w_rec);

-- initialize mean values
if opt.dataset=='Paris' or opt.dataset=='paris' then
	opt.mean_value1 = opt.paris_mean1
	opt.mean_value2 = opt.paris_mean2
	opt.mean_value3 = opt.paris_mean3
end
if opt.dataset=='ImageNet' or opt.dataset=='imagenet' then
	opt.mean_value1 = opt.imgnet20_mean1
	opt.mean_value2 = opt.imgnet20_mean2
	opt.mean_value3 = opt.imgnet20_mean3
end
print("mean_value:"..opt.mean_value1.." "..opt.mean_value2.." "..opt.mean_value3)

---------------------------------------------------------------------------
-- Initialize network variables
---------------------------------------------------------------------------

local featNet
if opt.featNet ~= nil and opt.featNet ~= '' then
	assert(paths.filep(opt.featNet),'featNet not found')
	featNet = util.load(opt.featNet, opt.gpu)
	assert(torch.type(featNet:get(#featNet.modules)) == 'nn.Linear')
	featNet:remove(#featNet.modules)
	featNet:evaluate()
end

local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local Convolution = nn.SpatialConvolution
local SBatchNorm = nn.SpatialBatchNormalization
local function basicblock(nInputPlane, nOutputPlane, stride)
    local block = nn.Sequential()
    local s = nn.Sequential()
    s:add(SBatchNorm(nInputPlane))
    s:add(nn.ReLU(true))
    s:add(Convolution(nInputPlane,nOutputPlane,3,3,stride,stride,1,1))
    s:add(SBatchNorm(n))
    s:add(nn.ReLU(true))
    s:add(Convolution(nOutputPlane,nOutputPlane,3,3,1,1,1,1))
    
    return block
        :add(nn.ConcatTable()
            :add(s)
            :add(nn.Identity()))
        :add(nn.CAddTable(true))
end

local nc = opt.nc
local nz = opt.nz
local nBottleneck = opt.nBottleneck
local ndf = opt.ndf
local ngf = opt.ngf
local nef = opt.nef
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

---------------------------------------------------------------------------
-- Generator net
---------------------------------------------------------------------------
-- Encode Input Context to noise (architecture similar to Discriminator)
local netE = nn.Sequential()
-- input is (nc) x 128 x 128
netE:add(SpatialConvolution(nc, nef, 4, 4, 2, 2, 1, 1))
netE:add(nn.ReLU(true))
-- state size: (nef) x 64 x 64
netE:add(SpatialConvolution(nef, nef, 3, 3, 1, 1, 1, 1))
netE:add(SpatialBatchNormalization(nef)):add(nn.ReLU(true))
-- state size: (nef) x 64 x 64
netE:add(SpatialConvolution(nef, nef * 2, 4, 4, 2, 2, 1, 1))
netE:add(SpatialBatchNormalization(nef * 2)):add(nn.ReLU(true))
-- state size: (nef*2) x 32 x 32
netE:add(SpatialConvolution(nef * 2, nef * 2, 3, 3, 1, 1, 1, 1))
netE:add(SpatialBatchNormalization(nef * 2)):add(nn.ReLU(true))
-- state size: (nef*2) x 32 x 32
netE:add(SpatialConvolution(nef * 2, nef * 4, 4, 4, 2, 2, 1, 1))
netE:add(SpatialBatchNormalization(nef * 4)):add(nn.ReLU(true))
-- state size: (nef*4) x 16 x 16
netE:add(SpatialConvolution(nef * 4, nef * 4, 3, 3, 1, 1, 1, 1))
netE:add(SpatialBatchNormalization(nef * 4)):add(nn.ReLU(true))
-- state size: (nef*4) x 16 x 16
netE:add(SpatialConvolution(nef * 4, nef * 8, 3, 3, 1, 1, 1, 1))
netE:add(SpatialBatchNormalization(nef * 8)):add(nn.ReLU(true))
-- state size: (nef*8) x 16 x 16
netE:add(SpatialConvolution(nef * 8, nef * 4, 3, 3, 1, 1, 1, 1))
netE:add(SpatialBatchNormalization(nef * 4)):add(nn.ReLU(true))
-- state size: (nef*4) x 16 x 16

local netG = nn.Sequential()
netG:add(netE)
-- state size: (ngf*4) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 4, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 32 x 32
netG:add(SpatialConvolution(ngf * 4, ngf * 2, 3, 3, 1, 1, 1, 1))
netG:add(SpatialBatchNormalization(nef * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 32 x 32
netG:add(SpatialFullConvolution(ngf * 2, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 64 x 64
netG:add(SpatialConvolution(ngf * 2, ngf, 3, 3, 1, 1, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 64 x 64
netG:add(SpatialFullConvolution(ngf, ngf/2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf/2)):add(nn.ReLU(true))
-- state size: (ngf/2) x 128 x 128
netG:add(SpatialConvolution(ngf/2, 3, 3, 3, 1, 1, 1, 1))
netG:add(nn.Tanh())
-- state size: (3) x 128 x 128

netG:apply(weights_init)
if opt.loadNetG ~= nil and opt.loadNetG ~= '' then
	netG = util.load(opt.loadNetG, opt.gpu)
	netG:training()
end

---------------------------------------------------------------------------
-- Adversarial discriminator net
---------------------------------------------------------------------------
local netD = nn.Sequential()
-- input is (nc) x 64 x 64, going into a convolution
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)
if opt.loadNetD ~= nil and opt.loadNetD ~= '' then
	netD = util.load(opt.loadNetD, opt.gpu)
	netD:training()
end

---------------------------------------------------------------------------
-- Loss Metrics
---------------------------------------------------------------------------
local criterion = nn.BCECriterion()
local criterionMSE
if opt.wtl2~=0 then
  criterionMSE = nn.MSECriterion()
end
local featureMSE
if opt.featNet~='' then
  featureMSE = nn.MSECriterion()
end

---------------------------------------------------------------------------
-- Setup Solver
---------------------------------------------------------------------------
print('LR of Gen is ',(opt.wtl2>0 and opt.wtl2<1) and 10 or 1,'times Adv')
optimStateG = {
   learningRate = (opt.wtl2>0 and opt.wtl2<1) and opt.lr*10 or opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

---------------------------------------------------------------------------
-- Initialize data variables
---------------------------------------------------------------------------
local input_ctx_vis = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local input_ctx = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local input_center = torch.Tensor(opt.batchSize, nc, opt.fineSize/2, opt.fineSize/2)
local input_real_center
if opt.wtl2~=0 then
    input_real_center = torch.Tensor(opt.batchSize, nc, opt.fineSize/2, opt.fineSize/2)
end

local real_img
local fake_img
local input_img
local real_feat
local fake_feat
if opt.featNet~=nil and opt.featNet~='' then
	real_img = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
	fake_img = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
	input_img = torch.Tensor(opt.batchSize, nc, 224, 224)
end
local df_dg_feat_crop = torch.Tensor(opt.batchSize, opt.nc, opt.fineSize/2, opt.fineSize/2):fill(0)
local df_dg_pad = torch.Tensor(opt.batchSize, opt.nc, opt.fineSize, opt.fineSize):fill(0)

local label = torch.Tensor(opt.batchSize)
local errD, errG, errG_l2, errG_feat
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

if pcall(require, 'cudnn') and pcall(require, 'cunn') and opt.gpu>0 then
    print('Using CUDNN !')
end
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input_ctx_vis = input_ctx_vis:cuda(); input_ctx = input_ctx:cuda();  input_center = input_center:cuda()
   label = label:cuda()
   netG = util.cudnn(netG);     netD = util.cudnn(netD)
   netD:cuda();           netG:cuda();           criterion:cuda();      
   if opt.wtl2~=0 then
      criterionMSE:cuda(); input_real_center = input_real_center:cuda();
   end

   if featNet~=nil and featNet~='' then
      real_img = real_img:cuda(); fake_img = fake_img:cuda(); input_img = input_img:cuda()
      featNet = util.cudnn(featNet)
	  featNet:cuda()
	  featureMSE:cuda()
   end
   df_dg_feat_crop = df_dg_feat_crop:cuda()
   df_dg_pad = df_dg_pad:cuda()
end
print('NetG:',netG)
print('NetD:',netD)
--if featNet~=nil then print('featNet:',featNet) end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

---------------------------------------------------------------------------
-- Define generator and adversary closures
---------------------------------------------------------------------------
-- create closure to evaluate f(X) and df/dX of discriminator

local fDx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real_ctx = data:getBatch()
   real_img:copy(real_ctx)
   fake_img:copy(real_ctx)

   local real_center = real_ctx[{{},{},{1 + opt.fineSize/4, opt.fineSize/2 + opt.fineSize/4},{1 + opt.fineSize/4, opt.fineSize/2 + opt.fineSize/4}}]:clone() -- copy by value
   real_ctx[{{},{1},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = opt.mean_value1 * 2.0 - 1.0
   real_ctx[{{},{2},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = opt.mean_value2 * 2.0 - 1.0
   real_ctx[{{},{3},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = opt.mean_value3 * 2.0 - 1.0

   real_img:add(1.0):mul(0.5)
   fake_img:add(1.0):mul(0.5)
   --mean = { 0.485, 0.456, 0.406 }
   --std = { 0.229, 0.224, 0.225 }
   input_img[{{},{1},{},{}}]:fill( opt.featNet_mean1 )
   input_img[{{},{2},{},{}}]:fill( opt.featNet_mean2 )
   input_img[{{},{3},{},{}}]:fill( opt.featNet_mean3 )
   input_img[{{},{},{1 + (224-opt.fineSize)/2, (224+opt.fineSize)/2},{1 + (224-opt.fineSize)/2, (224+opt.fineSize)/2}}]:copy(real_img)
   input_img[{{},{1},{},{}}]:add( -opt.featNet_mean1 ):div( opt.featNet_std1 )
   input_img[{{},{2},{},{}}]:add( -opt.featNet_mean2 ):div( opt.featNet_std2 )
   input_img[{{},{3},{},{}}]:add( -opt.featNet_mean3 ):div( opt.featNet_std3 )
   real_feat = featNet:forward(input_img):clone()
   data_tm:stop()

   input_ctx:copy(real_ctx)
   input_center:copy(real_center)
   if opt.wtl2~=0 then
      input_real_center:copy(real_center)
   end
   label:fill(real_label)

   local output = netD:forward(input_center)
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input_center, df_do)
   
   -- train with fake
   local fake = netG:forward(input_ctx)
   input_center:copy(fake[{{},{},{1 + opt.fineSize/4, opt.fineSize*3/4},{1 + opt.fineSize/4, opt.fineSize*3/4}}])
   label:fill(fake_label)

   data_tm:resume()
   local fake_clone = fake:clone()
   fake_clone:add(1.0):mul(0.5)
   fake_img[{{},{},{1 + opt.fineSize/4, opt.fineSize*3/4},{1 + opt.fineSize/4, opt.fineSize*3/4}}]:copy(
      fake_clone[{{},{},{1 + opt.fineSize/4, opt.fineSize*3/4},{1 + opt.fineSize/4, opt.fineSize*3/4}}])
   input_img[{{},{1},{},{}}]:fill( opt.featNet_mean1 )
   input_img[{{},{2},{},{}}]:fill( opt.featNet_mean2 )
   input_img[{{},{3},{},{}}]:fill( opt.featNet_mean3 )
   input_img[{{},{},{1 + (224-opt.fineSize)/2, (224+opt.fineSize)/2},{1 + (224-opt.fineSize)/2, (224+opt.fineSize)/2}}]:copy(fake_img)
   input_img[{{},{1},{},{}}]:add( -opt.featNet_mean1 ):div( opt.featNet_std1 )
   input_img[{{},{2},{},{}}]:add( -opt.featNet_mean2 ):div( opt.featNet_std2 )
   input_img[{{},{3},{},{}}]:add( -opt.featNet_mean3 ):div( opt.featNet_std3 )
   fake_feat = featNet:forward(input_img):clone()
   data_tm:stop()

   local output = netD:forward(input_center)
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input_center, df_do)

   errD = errD_real + errD_fake

   real_ctx = nil; real_center = nil
   errD_real = nil; errD_fake = nil;
   fake = nil; fake_clone = nil;
   output = nil
   df_do = nil
   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward({input_ctx,noise})
   input_center:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward({input_ctx,input_center}) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput(input_center, df_do)

   -- use feature loss here
   errG_feat = featureMSE:forward(fake_feat, real_feat)
   local df_do_feat = featureMSE:backward(fake_feat, real_feat)
   local df_dg_feat = featNet:backward(input_img, df_do_feat)
   df_dg_feat_crop:fill(0)
   df_dg_feat_crop:copy(df_dg_feat[{{},{},{1 + (224-opt.fineSize/2)/2, (224+opt.fineSize/2)/2},{1 + (224-opt.fineSize/2)/2, (224+opt.fineSize/2)/2}}])

   local errG_total = errG * opt.w_adv + errG_feat * opt.w_feat

   if opt.wtl2~=0 then
      errG_l2 = criterionMSE:forward(input_center, input_real_center)
      local df_dg_l2 = criterionMSE:backward(input_center, input_real_center)

      local overlapL2Weight = 10
      local wtl2Matrix = df_dg_l2:clone():fill(overlapL2Weight * opt.w_rec)
	  wtl2Matrix[{{},{},{1 + opt.overlapPred, opt.fineSize/2 - opt.overlapPred},{1 + opt.overlapPred, opt.fineSize/2 - opt.overlapPred}}]:fill(opt.w_rec)
      -- use feature loss here
      df_dg_feat_crop:mul(opt.w_feat)
      df_dg:mul(opt.w_adv)
      df_dg:add(df_dg_feat_crop)
      df_dg:addcmul(1, wtl2Matrix, df_dg_l2)
      errG_total = errG + errG_l2 * opt.w_rec

   end

   df_dg_pad:fill(0)
   df_dg_pad[{{},{},{1 + opt.fineSize/4, opt.fineSize*3/4},{1 + opt.fineSize/4, opt.fineSize*3/4}}]:copy(df_dg)
   netG:backward(input_ctx, df_dg_pad)

   return errG_total, gradParametersG
end

---------------------------------------------------------------------------
-- Train Context Encoder
---------------------------------------------------------------------------
for epoch = opt.begin_epoch, opt.niter do
   epoch_tm:reset()
   --opt.ntrain = 320
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)
      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. 'Err_G_f: %.4f  Err_G_L2: %.4f   Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real, errG_feat or -1, errG_l2 or -1,
                 errG and errG or -1, errD and errD or -1))
      end
   end

   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil

   --if epoch-opt.begin_epoch < 5 or epoch % 20 == 0 then
   if epoch % 20 == 0 then
	  local modelG = 'checkpoints/'..opt.name..'_'..epoch..'_net_G.t7'
	  local modelD = 'checkpoints/'..opt.name..'_'..epoch..'_net_D.t7'
	  local logName = 'log_'..opt.name..'_'..epoch
      util.save(modelG, netG, opt.gpu)
      util.save(modelD, netD, opt.gpu)
   end

   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, opt.niter, epoch_tm:time().real))
end
