from vuer import Vuer, VuerSession
from utils.pose_streamer import FiducialPoseStreamerSE3
from vuer.schemas import Sphere, DefaultScene, TriMesh
from asyncio import sleep
import numpy as np
from killport import kill_ports

kill_ports(ports=[8012])

import trimesh

# scene_mesh = "/Users/alanyu/Library/CloudStorage/Dropbox/spark_aria/real/stata_kitchen_v1/vol_fusion_v1/mesh_272.ply"
scene_mesh = "/Users/alanyu/Library/CloudStorage/Dropbox/spark_aria/real/spot_room_v1/vol_fusion_v1/mesh_234.ply"
tag_aria_tfs = "/Users/alanyu/Library/CloudStorage/Dropbox/spark_aria/real/spot_room_v1/vol_fusion_v1/sg_obs/world_tfs.npy"

tag_aria_tfs = np.load(tag_aria_tfs)
tag_aria_tf = tag_aria_tfs[0]

mesh = trimesh.load(scene_mesh)
vertices = mesh.vertices
faces = mesh.faces

app = Vuer()

target_point = [-0.8074752,  0.18065326, -0.50076877]
best_dir     = [0.176, 0.,    0.984]
# standoff_point = [-0.719,  0.181, -0.009] # 0.5
standoff_point = [-0.755, 0.181, -0.205] # 0.3 
# standoff_point = [-0.589,  0.064, -0.134]

# target_point = [1.5,  -1.0, -0.50076877]
# standoff_point = [1.606, -0.53, -0.368]



@app.spawn(start=True)
async def main(sess: VuerSession):
    sess.set @ DefaultScene(
        TriMesh(vertices=vertices, faces=faces, key="scene"),
        Sphere(
            args=[0.1, 20, 20],
            position=target_point,
            material=dict(color="blue"),
            key="target",
        ),
        Sphere(
            args=[0.1, 20, 20],
            position=standoff_point,
            material=dict(color="red"),
            key="standoff",
        ),
    )


    while True:
        await sleep(0.1)

