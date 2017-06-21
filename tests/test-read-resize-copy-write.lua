scene_path=sim.getStringParameter(sim.stringparam_scene_path)
im={simIM.read(scene_path..'/lena.bmp')}
w,h=simIM.size(im[1])
w2=w
n=9
for i=1,n do
    im[i+1]=simIM.resize(im[1],w/2^i,h/2^i,simIM.interp.area)
    w2=w2+w/2^i
end
im2=simIM.create(w2,h)
x=0
for i=0,n do
    for j=0,2^i-1 do
        y=j*h/2^i
	simIM.copy(im[i+1],0,0,im2,x,y,w/2^i,h/2^i)
    end
    x=x+w/2^i
end
simIM.write(im2,scene_path..'/tiled-lena.jpg')
for i=1,#im do simIM.destroy(im[i]) end
simIM.destroy(im2)
print(simIM.numActiveHandles()..' active handles')
