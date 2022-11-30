scene_path=sim.getStringParameter(sim.stringparam_scene_path)
im=simIM.read(scene_path..'/rgb1.png')
sz=simIM.size(im)
for i=1,2000 do
    p={math.random(sz[1]),math.random(sz[2])}
    v=simIM.get(im,p)
    for i=1,3 do v[i]=255-v[i] end
    simIM.set(im,p,v)
end
simIM.write(im,scene_path..'/test-set-get.png')
simIM.destroy(im)
--print(simIM.numActiveHandles()..' active handles')
