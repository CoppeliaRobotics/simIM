scene_path=sim.getStringParameter(sim.stringparam_scene_path)
im=simIM.read(scene_path..'/lena.bmp')
sz=simIM.size(im)
w=sz[1];h=sz[2]

p0={w/2,h/2}
for i=1,12 do
    angle=math.pi*2*i/12
    p1={p0[1]+math.cos(angle)*w/3,p0[1]+math.sin(angle)*h/3}
    simIM.line(im,p0,p1,{255,255,0},i,16)
end

simIM.write(im,scene_path..'/test-lines.jpg')
simIM.destroy(im)
print(simIM.numActiveHandles()..' active handles')
