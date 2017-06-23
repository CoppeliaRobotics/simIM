scene_path=sim.getStringParameter(sim.stringparam_scene_path)
im=simIM.create(320,200)
p1={-100,-50}
p2={430,190}
print(string.format('%d,%d,%d,%d ->',p1[1],p1[2],p2[1],p2[2]))
valid,p1,p2=simIM.clipLine(im,p1,p2)
print(string.format('            -> %d,%d,%d,%d',p1[1],p1[2],p2[1],p2[2]))
simIM.line(im,p1,p2,{255,0,0},5,8,0)
simIM.write(im,scene_path..'/test-clipline.png')
simIM.destroy(im)
print(simIM.numActiveHandles()..' active handles')
