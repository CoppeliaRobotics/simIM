scene_path=sim.getStringParameter(sim.stringparam_scene_path)
function text(f)
    return string.format('(%s) %s', f, 'The quick brown fox jumps over the lazy dog')
end
lineSpacing=8
w=0
h=lineSpacing
for k,v in pairs(simIM.fontFace) do
    w1,h1,bl=simIM.textSize(text(k),v,false)
    w=math.max(w,w1)
    h=h+h1+bl+lineSpacing
end
im=simIM.create(w,h)
y=0
for k,v in pairs(simIM.fontFace) do
    w1,h1,bl=simIM.textSize(text(k),v,false)
    y=y+h1
    simIM.text(im,text(k),0,y,v,false)
    y=y+bl+lineSpacing
end

simIM.write(im,scene_path..'/test-text.jpg')
simIM.destroy(im)
