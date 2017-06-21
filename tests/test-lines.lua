scene_path=sim.getStringParameter(sim.stringparam_scene_path)
im=simIM.read(scene_path..'/lena.bmp')
w,h=simIM.size(im)

x0=w/2
y0=h/2
for i=1,12 do
    angle=math.pi*2*i/12
    x1=x0+math.cos(angle)*w/3
    y1=y0+math.sin(angle)*h/3
    simIM.line(im, x0, y0, x1, y1, 255, 255, 0, i, 16)
end

simIM.write(im,scene_path..'/test-lines.jpg')
simIM.destroy(im)
