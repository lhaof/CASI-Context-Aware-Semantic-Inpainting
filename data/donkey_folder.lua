--[[
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).

    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]

    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

--require 'image'
local image = paths.dofile('own_image.lua')
paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS

-- Check for using edge
if opt.use_edge == 0 then
	print('no using edge.')
elseif opt.use_edge == 1 then
	print('using edge.')
	assert(opt.EDGE_ROOT ~= nil, 'EDGE_ROOT cannot be nil')
	assert(opt.EDGE_ROOT ~= '', 'EDGE_ROOT cannot be null string')
	assert(paths.dirp(opt.EDGE_ROOT), 'cannot find EDGE_ROOT: '..opt.EDGE_ROOT)
	print('Edge root: ',opt.EDGE_ROOT)
	opt.edge = opt.EDGE_ROOT
end

------------------------------------------

-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT')
if not paths.dirp(opt.data) then
    error('Did not find directory: ', opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local nc = opt.nc
local loadSize   = {nc, opt.loadSize}
local sampleSize = {nc, opt.fineSize}

local function loadImage(path)
	local input = image.load(path, nc, 'float')
	
	local edgeData
	if opt.use_edge == 1 then

		idx = 0; pathSplit = {};
		for x in string.gmatch(path,"[^/]*") do
			if x:len() > 0 then
				idx = idx + 1; pathSplit[idx] = x;
			end
		end
		-- for k,v in ipairs(pathSplit) do print(k,v) end

		local imageFile = pathSplit[idx]
		local edgeFile
		local edgeFileExt = ".jpg"
		for k,ext in ipairs({".jpg",".JPG",".jpeg",".JPEG",".png",".PNG"}) do
			if string.find(imageFile,ext) ~= nil then
				edgeFile = string.gsub(imageFile,ext,edgeFileExt)
				break
			end
		end	
		assert(edgeFile ~= nil, 'image file has wrong extension')

		local edgePath = paths.concat(opt.edge,edgeFile)

		assert(paths.filep(edgePath), 'edge map does not exist: '..edgePath)

		edgeData = image.load(edgePath, nc, 'float')
		-- print('edgeData.size: ',edgeData:size())
		-- print('input.size: ', input:size())

		if edgeData:size(1)~=input:size(1) or 
			edgeData:size(2)~=input:size(2) or
			edgeData:size(3)~=input:size(3) then
			error('edgemap size does not match image')
		end
	end

	if opt.use_edge == 1 then assert(edgeData ~= nil) end

	-- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
	if loadSize[2]>0 then
		local iW = input:size(3)
		local iH = input:size(2)
		if iW < iH then
			input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
		if opt.use_edge==1 then
			edgeData = image.scale(edgeData, loadSize[2], loadSize[2] * iH / iW)
		end
	else
		input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
		if opt.use_edge==1 then
			edgeData = image.scale(edgeData, loadSize[2] * iW / iH, loadSize[2])
		end
	end
	elseif loadSize[2]<0 then
		local scalef = 0
		if loadSize[2] == -1 then
			scalef = torch.uniform(0.5,1.5)
		else
			scalef = torch.uniform(1,3)
		end
		local iW = scalef*input:size(3)
		local iH = scalef*input:size(2)
		input = image.scale(input, iH, iW)
		if opt.use_edge==1 then
			edgeData = image.scale(edgeData, iH, iW)
		end
	end

	if opt.use_edge == 1 then
		assert(edgeData ~= nil)
		return input, edgeData
	else
		return input
	end
	return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
	collectgarbage()

	local input
	local input_edge
	if opt.use_edge == 1 then
		input, input_edge = loadImage(path)
	else
		input = loadImage(path)
	end
	
	if opt.use_edge == 1 then assert(input_edge ~= nil) end

	local iW = input:size(3)
	local iH = input:size(2)
	
	-- do random crop
	local oW = sampleSize[2];
	local oH = sampleSize[2]
	local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
	local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
	local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
	local out_edge
	if opt.use_edge == 1 then
		out_edge = image.crop(input_edge, w1, h1, w1 + oW, h1 + oH)
	end

	assert(out:size(2) == oW)
	assert(out:size(3) == oH)
	if opt.use_edge == 1 then
		assert(out_edge:size(2) == oW)
		assert(out_edge:size(3) == oH)
	end

	-- do hflip with probability 0.5
	if torch.uniform() > 0.5 then 
		out = image.hflip(out)
		if opt.use_edge == 1 then
			out_edge = image.hflip(out_edge)
		end
	end

	local out_feat
	if opt.featNet ~= nil and opt.featNet ~= '' then
		out_feat = image.scale(out, 224, 224)
	end

	-- make it [0, 1] -> [-1, 1]
	out:mul(2):add(-1)
	--[[ no need for edge
	if opt.use_edge == 1 then
		out_edge:mul(2):add(-1)
	end
	--]]

	if opt.use_edge == 1 then
		return out, out_edge
	elseif opt.featNet ~= nil and opt.featNet ~= '' then
		return out, out_feat
	else
		return out
	end
end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   if opt.quiet~=1 then print('Loading train metadata from cache') end
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {nc, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {nc, sampleSize[2], sampleSize[2]}
else
   if opt.quiet~=1 then print('Creating train metadata') end
   trainLoader = dataLoader{
      paths = {opt.data},
      loadSize = {nc, loadSize[2], loadSize[2]},
      sampleSize = {nc, sampleSize[2], sampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   if opt.quiet~=1 then print('saved metadata cache at', trainCache) end
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end
