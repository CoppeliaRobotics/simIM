scene_path=sim.getStringParameter(sim.stringparam_scene_path)
im=simIM.create(800,600)
simIM.line(im,10,10,20,40,0,255,0,1,8,0)
simIM.arrowedLine(im,20,40,70,30,0,255,0,1,8,0,0.2)
simIM.fillPoly(im,{50,100,150,200,100,300,150,100,100,200,150,300},{3,3},55,55,0)
simIM.polylines(im,{50,100,150,200,100,300,150,100,100,200,150,300},{3,3},true,0,0,255)
simIM.rectangle(im,10,300,50,360,0,55,0,-1)
simIM.rectangle(im,10,300,50,360,50,255,50,4,4)
simIM.circle(im,200,300,50,255,0,255,-1)
simIM.circle(im,400,300,50,0,255,255,1)
simIM.ellipse(im,600,300,50,100,0,0,270,255,255,0,1)
simIM.write(im,scene_path..'/test-drawing.png')
simIM.destroy(im)
print(simIM.numActiveHandles()..' active handles')
