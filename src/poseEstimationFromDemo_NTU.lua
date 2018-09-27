require 'paths'
require 'torch'
require 'xlua'
require 'nn'
require 'nnx'
require 'nngraph'
require 'image'
require 'hdf5'
require 'sys'
require 'lfs'
require 'cunn'
require 'cutorch'
require 'cudnn'

paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')
paths.dofile('util/pyTools.lua')


function lines_from(file)
  lines_ = {}
  for line in io.lines(file) do
    lines_[#lines_ + 1] = line
  end
  return lines_
end


local matio = require 'matio'

local J = 16
local inputRes = 256
local outputRes = 64

-- it is important that the person should be in the center of the image 
-- and most of the body parts should be within the image

local cropped_image_set = {}
local points_set = {}
local model = torch.load('../models/hgreg-3d.t7')   -- load the pretrained model

local txt_file = '/mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/NTU/cropImg_for3Dpose/S014/S014_action_train.txt'
data_dir = "test"
result_dir_keyword = "PoseTowards3D_allRGB_2classes"


local lines_ = lines_from(txt_file)

for lineid, linecontent in pairs(lines_) do
  -- print(linecontent)
  --local line_split = split(linecontent)
  -- for i in string.gmatch(linecontent, "%S+") do
    -- img_dir = i
	-- break     -- only take out the first split
  -- end
  words = {}
  for word in string.gmatch(linecontent, "%S+") do table.insert(words, word) end
  img_dir = words[1]
  -- classid = words[2]
  local handle = assert(io.popen('ls -1v ' .. img_dir))
  local allFileNames = string.split(assert(handle:read('*a')), '\n')
  local numImages = #allFileNames
  print(string.format("%d image fiels found in the given directory.", numImages))	
  -- print(allFileNames)
  result_dir = string.gsub(img_dir, "cropImg_for3Dpose", "estimated3Dpose")         -- switch 'cropImg_for3Dpose'  with   'estimated3Dpose'
  
  --result_dir = string.gsub(result_dir, "Activity/RGB_Laptop1", "ActivityRecog/4classes/Pose3D_Laptop1")    -- switch 'RGB' with  'PoseTowards3D_allRGB_2classes'
  -- /mnt/Projects/CV-008_Students/ActionRecognitionProjects/TrainingData/Airport_Lin/ActivityRecog/4classes/Pose3D_5735 
  --result_dir = string.format("%s_%d",result_dir, classid)
  print(result_dir)
  if not paths.dirp(result_dir) then
    os.execute("mkdir " .. result_dir)   -- mkdir can create new folder in only one layer, not in several layers
  end
  
  for fileid, filename in pairs(allFileNames) do
	  if filename:find('png' .. '$') then
		img_path = paths.concat(img_dir, filename)
		io.write("image path: ")
		io.write(img_path .. "\n")
		local img = image.load(img_path):narrow(1, 1, 3)  -- take the segment of the original image, in dimension 1 from index 1 to index 3 (take the first 3 rows),  the first dimension is the RGB channel!!!
		local h, w = img:size(2), img:size(3)   -- size(2) height   size(3) width   size(1) RGB channel 
		local c = torch.Tensor({w / 2, h / 2})
		local size = math.max(h, w)
		local inp = crop(img, c, 1 * size / 200.0, 0, inputRes)    -- function crop(img, center, scale, rot, res) defined in img.lua line 168
		
		print(string.format("shape of original image %d %d", img:size(2),img:size(3)))     
		print(string.format("shape of cropped image %d %d", inp:size(2),inp:size(3)))   -- 256 * 256                                                         -- inputRes = 256
		

		local output = model:forward(inp:view(1, 3, inputRes, inputRes):cuda())
		-- print(string.format("number of output: %d", #output))
		-- print(output)
		  -- 1 : CudaTensor - size: 1x16x64x64
		  -- 2 : CudaTensor - size: 1x16x64x64   tmpOutput
		  -- 3 : CudaTensor - size: 1x16         Reg
		local output_1 = output[1]
		-- print("output_1=")
		-- print(output_1[1][1])
		  
		local tmpOutput = output[#output - 1]   -- 2 : CudaTensor - size: 1x16x64x64
		-- print("output_2=")
		-- print(tmpOutput[1][1])
		
		local p = getPreds(tmpOutput)   -- 1*16*2
		-- print("p=")
		-- print(p)
		local Reg = output[#output]  -- 3 : CudaTensor - size: 1x16
		-- print("Reg=")
		-- print(Reg)
		local z = (Reg + 1) * outputRes / 2    -- outputRes = 64    z: 1*16
		local pred = torch.zeros(J, 3)     -- pred : 16 * 3 

		for j = 1, J do     --  J = 16 
			local hm = tmpOutput[1][j]
			local pX, pY = p[1][j][1], p[1][j][2]
			if pX > 1 and pX < outputRes and pY > 1 and pY < outputRes then
				local diff = torch.FloatTensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
				p[1][j]:add(diff:sign():mul(.25))
			end
		end
		p:add(0.5)
		
		-- print("p=")
		-- print(p)

		for j = 1, J do 
			pred[j][1], pred[j][2], pred[j][3] = p[1][j][1], p[1][j][2], z[1][j]
		end
		
		-- print("pred=")
		-- print(pred)

		pred = pred * 4
		cropped_image_set[filename]=inp
		points_set[filename]=pred
		
		--pyFunc('Show3d', {joint=pred, img=inp, point2D=p, noPause = torch.zeros(1)})
		
		matio.save(string.format(result_dir .. '/Points%d.mat',fileid ),pred)
		--matio.save(string.format(result_dir .. 'Img%d.mat',fileid ),inp)
		
		
		-- local data = {points=pred, cropped_image=inp}
		-- local tmpFile = 'tmp/ScanPassenger.h5'
		-- if io.open(tmpFile) then os.execute('rm ' .. tmpFile) end
		-- saveData(data, tmpFile)
	  end
		
  end
  -- print(line_split)
  --local dirname = line_split[1]
  --print(dirname)
end

-- local counter = 0
-- for filename in paths.files(img_dir) do
  -- --print(paths.thisfile(filename))
  -- if filename:find('png' .. '$') then
    -- print(paths.concat(img_dir, filename))
	-- counter = counter + 1
  -- end
-- end
-- print(counter)


-- for fileid, filename in pairs(allFileNames) do
  -- if filename:find('png' .. '$') then
    -- print(paths.concat(img_dir, filename))
  -- end
-- end



-- local counter = "0"

-- print(#cropped_image_set)
-- print(#points_set)


-- matio.save('tmp/ScanPassengerPoints.mat', points_set)

   -- pyFunc(func, data) -- data is a dict
-- pyFunc('Show3dSequence', {points_set=points_set, cropped_image_set=cropped_image_set})   -- pyFunc(func, data) -- data is a dict

-- local tmpFile = 'tmp/' .. torch.random() ..  '.h5'


-- local data = {points_set=points_set, cropped_image_set=cropped_image_set}
-- saveData(data, tmpFile)
