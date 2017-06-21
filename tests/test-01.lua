scene_path=sim.getStringParameter(sim.stringparam_scene_path)
im=simIM.read(scene_path..'/lena.bmp')
w,h=simIM.size(im)

simIM.addWeighted(im, im, -1, 0, 255, true)

simIM.write(im,scene_path..'/test-01.jpg')
simIM.destroy(im)
