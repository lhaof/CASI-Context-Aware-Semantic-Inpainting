own_image = {}
local image_lib = require 'image'
function own_image.load(path, channels, data_type)
	return image_lib.load(path, channels, data_type)
end
function own_image.scale(input, width, height)
	return image_lib.scale(input, width, height)
end
function own_image.crop(input, w1, h1, w2, h2)
	return image_lib.crop(input, w1, h1, w2, h2)
end
function own_image.hflip(input)
	return image_lib.hflip(input)
end
return own_image
