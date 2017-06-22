scene_path=sim.getStringParameter(sim.stringparam_scene_path)
im=simIM.read(scene_path..'/rgb1.png')
w,h=simIM.size(im)
for i=1,2000 do
    x=math.random(w)
    y=math.random(h)
    p=simIM.get(im,x,y)
    for i=1,3 do p[i]=255-p[i] end
    simIM.set(im,x,y,p)
end
simIM.write(im,scene_path..'/test-set-get.png')
simIM.destroy(im)
print(simIM.numActiveHandles()..' active handles')
