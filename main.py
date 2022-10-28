from turtle import width
import mujoco_py
import os
from PIL import Image
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = "/home/larissa/Documents/TCC/simulacao/content/environment.xml"
print(xml_path)
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)


print([(x.geom1,x.geom2) for x in sim.data.contact])


# win = mujoco_py.MjViewer(sim)


# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]



sim.step() 
cam_img = sim.render(width=800,height=600,camera_name="depth_camera1")
Image.fromarray(cam_img[::-1]).show()
print(sim.data.qpos)

input()