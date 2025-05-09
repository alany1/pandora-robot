###########################################################################
#  spot_fiducial_pose.py
#
#  Two convenience helpers:
#    • FiducialPoseStreamerSE2  – (x, y, yaw) in the tag’s filtered frame
#    • FiducialPoseStreamerSE3  – full 4×4 homogeneous transform
#
#  Read-only: no lease required.
###########################################################################

import time
from typing import Tuple, Generator

import numpy as np
import bosdyn.client, bosdyn.client.util
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    BODY_FRAME_NAME,
    get_se2_a_tform_b,
    get_a_tform_b,
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.api import world_object_pb2 as wo_pb2


# ------------------------------------------------------------------ helpers
def _wait_for_tag(obj_client: WorldObjectClient, tag_id: int):
    """Block until the specified AprilTag is visible; return its object + frame name."""
    while True:
        for wo in obj_client.list_world_objects(
                object_type=[wo_pb2.WORLD_OBJECT_APRILTAG]).world_objects:
            if wo.apriltag_properties.tag_id == tag_id:
                fid_frame = wo.apriltag_properties.frame_name_fiducial_filtered
                return wo, fid_frame
        print(f"[FidPose] Waiting to see tag {tag_id} …")
        time.sleep(0.5)


# ===============================================================  SE(2)  ==
class FiducialPoseStreamerSE2:
    """Stream (x, y, yaw) of gravity-aligned body in the fiducial frame."""

    def __init__(self, robot_ip: str, tag_id: int,
                 sdk_name: str = "FidPoseClientSE2") -> None:
        sdk   = bosdyn.client.create_standard_sdk(sdk_name)
        robot = sdk.create_robot(robot_ip)
        bosdyn.client.util.authenticate(robot)
        robot.time_sync.wait_for_sync()

        self._state = robot.ensure_client(RobotStateClient.default_service_name)
        wo_client   = robot.ensure_client(WorldObjectClient.default_service_name)

        tag_obj, fid_frame = _wait_for_tag(wo_client, tag_id)
        odom_T_fid         = get_se2_a_tform_b(tag_obj.transforms_snapshot,
                                               ODOM_FRAME_NAME, fid_frame)
        self._fid_T_odom   = odom_T_fid.inverse()   # cache
        print(f"[SE2] Using frame {fid_frame}")

    # ---------------------------------------------------------------- API --
    def body_pose(self) -> Tuple[float, float, float]:
        """Return (x, y, yaw)."""
        snap        = self._state.get_robot_state().kinematic_state.transforms_snapshot
        odom_T_body = get_se2_a_tform_b(snap, ODOM_FRAME_NAME,
                                        GRAV_ALIGNED_BODY_FRAME_NAME)
        fid_T_body  = self._fid_T_odom * odom_T_body
        return fid_T_body.x, fid_T_body.y, fid_T_body.angle

    def stream_forever(self, hz: float = 10.0) -> None:
        dt = 1.0 / hz
        try:
            while True:
                x, y, yaw = self.body_pose()
                print(f"x={x:+.3f}  y={y:+.3f}  yaw={yaw*180/3.1416:+.1f}°")
                time.sleep(dt)
        except KeyboardInterrupt:
            print("\n[SE2 stream stopped]")

import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy as np
from bosdyn.client.math_helpers import SE3Pose, Quat          # whatever path you use

def mean_se3(pose_list):
    """
    Average a list of math_helpers.SE3Pose objects.
    Returns a new SE3Pose (same class).

    Notes
    -----
    • All quaternions are first flipped to the same hemisphere so the
      mean is well-defined.
    • Works well as long as the rotations are not scattered over >180 °.
    """
    if len(pose_list) == 0:
        raise ValueError("pose_list is empty")

    # -------- 1. mean translation -----------------------------------------
    xyz = np.array([[p.x, p.y, p.z] for p in pose_list])
    t_bar = xyz.mean(axis=0)

    # -------- 2. mean rotation (Markley) ----------------------------------
    M = np.zeros((4, 4))
    for p in pose_list:
        q = np.array([p.rot.w, p.rot.x, p.rot.y, p.rot.z], dtype=float)
        if q[0] < 0:                          # enforce hemisphere
            q = -q
        M += np.outer(q, q)

    # principal (largest-eigenvalue) eigenvector  →  mean quaternion
    eigvals, eigvecs = np.linalg.eigh(M)
    q_bar = eigvecs[:, eigvals.argmax()]
    if q_bar[0] < 0:                          # optional: keep w ≥ 0
        q_bar = -q_bar
    q_bar = Quat(q_bar[0], q_bar[1], q_bar[2], q_bar[3])

    # -------- 3. assemble -------------------------------------------------
    return SE3Pose(t_bar[0], t_bar[1], t_bar[2], q_bar)


# ===============================================================  SE(3)  ==
class FiducialPoseStreamerSE3:
    """
    Stream full 4×4 transform (fiducial → body).  Handy for robotics math.
    """

    def __init__(self, robot_ip: str, tag_id: int,
                 sdk_name: str = "FidPoseClientSE3") -> None:
        sdk   = bosdyn.client.create_standard_sdk(sdk_name)
        robot = sdk.create_robot(robot_ip)
        bosdyn.client.util.authenticate(robot)
        robot.time_sync.wait_for_sync()

        self._state = robot.ensure_client(RobotStateClient.default_service_name)
        wo_client   = robot.ensure_client(WorldObjectClient.default_service_name)

        tag_obj, fid_frame = _wait_for_tag(wo_client, tag_id)
        tfs = []
        for _ in range(100):
            odom_T_fid         = get_a_tform_b(tag_obj.transforms_snapshot,
                                               ODOM_FRAME_NAME, fid_frame)
            tfs.append(odom_T_fid.inverse())
            
        print(len(tfs))
            
        # make avg
        self._fid_T_odom = mean_se3(tfs)
            
            
        
        # self._fid_T_odom   = odom_T_fid.inverse()
        print(f"[SE3] Using frame {fid_frame}")

    # ---------------------------------------------------------------- API --
    def body_pose(self):
        """Return bosdyn.client.math_helpers.SE3Pose."""
        snap        = self._state.get_robot_state().kinematic_state.transforms_snapshot
        odom_T_body = get_a_tform_b(snap, ODOM_FRAME_NAME,
                                    BODY_FRAME_NAME)
        return self._fid_T_odom * odom_T_body

    def body_transform_matrix(self) -> np.ndarray:
        """Return 4×4 homogeneous transform as np.ndarray (fiducial → body)."""
        return np.array(self.body_pose().to_matrix())

    def stream_matrices(self, hz: float = 10.0) -> Generator[np.ndarray, None, None]:
        """
        Yield 4×4 np.ndarray transforms forever (Ctrl-C to stop).
        Example:
            for T in se3.stream_matrices(5):
                print(T)
        """
        dt = 1.0 / hz
        try:
            while True:
                yield self.body_transform_matrix()
                time.sleep(dt)
        except KeyboardInterrupt:
            return


# ============================================================ DEMO block ==
if __name__ == "__main__":
    ROBOT_IP = "192.168.80.3"
    TAG_ID   = 16

    # --- SE2 demo --------------------------------------------------------
    se2 = FiducialPoseStreamerSE2(ROBOT_IP, TAG_ID)
    print("Streaming SE2 for 3 samples:")
    for _ in range(3):
        print(se2.body_pose())
        time.sleep(0.5)

    # --- SE3 demo --------------------------------------------------------
    se3 = FiducialPoseStreamerSE3(ROBOT_IP, TAG_ID)
    print("\nStreaming one 4×4 transform:")
    T = se3.body_transform_matrix()
    print(T)
