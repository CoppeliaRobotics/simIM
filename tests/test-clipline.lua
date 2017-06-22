scene_path=sim.getStringParameter(sim.stringparam_scene_path)
im=simIM.create(320,200)
x1=-100
y1=-50
x2=430
y2=190
print(string.format('%d,%d,%d,%d ->',x1,y1,x2,y2))
valid,x1,y1,x2,y2=simIM.clipLine(im,x1,y1,x2,y2)
print(string.format('            -> %d,%d,%d,%d',x1,y1,x2,y2))
simIM.line(im,x1,y1,x2,y2,255,0,0,5,8,0)
simIM.write(im,scene_path..'/test-clipline.png')
simIM.destroy(im)
print(simIM.numActiveHandles()..' active handles')
