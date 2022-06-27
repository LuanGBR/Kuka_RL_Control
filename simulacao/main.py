import mujoco_py as mjp 
from mujoco_py import utils

import os
# xml_path = "simulacao/content/kr16_mujoco/kr16_2.urdf"
xml_path = "simulacao/content/kr16_mujoco/kr16_2.xml"
model = mjp.load_model_from_path(xml_path)
sim = mjp.MjSim(model)

fig = mjp.MjViewerBasic(sim)

while(1):
    fig.render()