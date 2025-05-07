from vuer import Vuer, VuerSession
from utils.pose_streamer import FiducialPoseStreamerSE3
from vuer.schemas import Sphere, DefaultScene, TriMesh
from asyncio import sleep
import numpy as np
from killport import kill_ports

kill_ports(ports=[8012])

import trimesh

streamer = FiducialPoseStreamerSE3(robot_ip="192.168.80.3", tag_id=16)
# streamer.stream_forever(hz=5)            # prints 5 Hz
T = streamer.body_transform_matrix()          # 4×4 NumPy array
print(T)

scene_mesh = "/Users/alanyu/Downloads/fused_mesh.ply"
tag_aria_tfs = "/Users/alanyu/Library/CloudStorage/Dropbox/spark_aria/real/spot_room_v1/vol_fusion_v1/sg_obs/world_tfs.npy"

tag_aria_tfs = np.load(tag_aria_tfs)
tag_aria_tf = tag_aria_tfs[0]

mesh = trimesh.load(scene_mesh)
vertices = mesh.vertices
faces = mesh.faces

app = Vuer()

ALIGN_PUPIL_TO_BD = np.array([[ 0, -1,  0, 0],
                              [1,  0,  0, 0],
                              [ 0,  0,  -1, 0],
                              [ 0,  0,  0, 1]], dtype=float)

target_point = [0, 3, 0]


@app.spawn(start=True)
async def main(sess: VuerSession):
    sess.set @ DefaultScene(
        TriMesh(vertices=vertices, faces=faces, key="scene"),
        Sphere(
            args=[0.1, 20, 20],
            position=tag_aria_tf[:3, -1].tolist(),
            material=dict(color="yellow"),
            key="fudicial",
        ),
        Sphere(
            args=[0.1, 20, 20],
            position=target_point,
            material=dict(color="blue"),
            key="target",
        ),
    )
    
    
    while True:
        robot_tag_tf = streamer.body_transform_matrix()
        robot_aria_tf = tag_aria_tf @ ALIGN_PUPIL_TO_BD @ robot_tag_tf
        point_to_robot = np.linalg.inv(robot_aria_tf) @ np.array([[*target_point, 1]]).T
        
        position = robot_aria_tf[:3, -1].tolist()
        print(point_to_robot)
        
        sess.upsert @ Sphere(args=[0.1, 20, 20], position=position, material=dict(color="red"), key="spot")
        await sleep(0.1)
        
