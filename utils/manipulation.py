import time

import numpy as np
from bosdyn.api import manipulation_api_pb2 as mp, geometry_pb2

def make_walk_request(odom_T_hand, frame="odom", standoff=0.10):
    """
    Build a WalkToObjectRayInWorld request that asks Spot to stand
    *standoff* metres behind the desired hand point.
    """
    req = mp.ManipulationApiRequest()
    walk = req.walk_to_object_ray_in_world         # one-of field

    walk.frame_name = frame

    # ---- ray end: the hand target in odom frame -------------------------
    walk.ray_end_rt_frame.CopyFrom(geometry_pb2.Vec3(
        x=odom_T_hand.x,
        y=odom_T_hand.y,
        z=odom_T_hand.z))

    # ---- ray start: a point *behind* the target so the planner knows
    #      which side of the object to stop on
    dir_vec = odom_T_hand.rot.transform_point(1, 0, 0)  # +X of hand
    # set z to zero and normalize
    d = np.array(dir_vec)
    d[2] = 0
    d = d / np.linalg.norm(d)
    print(d, dir_vec)
    walk.ray_start_rt_frame.CopyFrom(geometry_pb2.Vec3(
        x=odom_T_hand.x - standoff * dir_vec[0],
        y=odom_T_hand.y - standoff * dir_vec[1],
        z=odom_T_hand.z - standoff * dir_vec[2]))

    return req
# 
# from bosdyn.api import manipulation_api_pb2 as mp, geometry_pb2
# import numpy as np
# 
# 
# def make_walk_request(odom_T_hand,
#                       v_hat,                   # unit approach vector  (numpy [3])
#                       frame="odom",
#                       standoff=0.35):
#     """
#     Walk so the hand target (odom_T_hand) is reachable *and* the gripper’s +X
#     axis can line up with v_hat (your approach direction).  standoff is the
#     body-to-grasp clearance in metres.
#     """
#     req  = mp.ManipulationApiRequest()
#     walk = req.walk_to_object_ray_in_world          # one-of field
#     walk.frame_name = frame
#     walk.offset_distance.value = standoff
# 
#     # --- ray end: the point you want the tool-frame origin to reach -----------
#     walk.ray_end_rt_frame.CopyFrom(geometry_pb2.Vec3(
#         x=odom_T_hand.x,
#         y=odom_T_hand.y,
#         z=odom_T_hand.z))
# 
#     # --- ray start: 4× farther back along the hand +X direction ---------------
#     dir_vec = odom_T_hand.rot.transform_point(1, 0, 0)   # +X of hand
#     walk.ray_start_rt_frame.CopyFrom(geometry_pb2.Vec3(
#         x=odom_T_hand.x - 4 * dir_vec[0],
#         y=odom_T_hand.y - 4 * dir_vec[1],
#         z=odom_T_hand.z - 4 * dir_vec[2]))
# 
#     # ──────────────────────────────────────────────────────────────────────────
#     #                NEW bits ↓   — orientation constraint
#     # ──────────────────────────────────────────────────────────────────────────
#     align = walk.orientation_constraint.vector_alignment_with_tolerance
# 
#     # (1)  Which axis on the gripper should line up?  +X is “straight out”
#     align.axis_on_gripper_ewrt_gripper.CopyFrom(
#         geometry_pb2.Vec3(x=1, y=0, z=0))
# 
#     # (2)  What direction in the *walk.frame_name* should it line up with?
#     align.axis_to_align_with_ewrt_frame.CopyFrom(
#         geometry_pb2.Vec3(x=float(v_hat[0]),
#                           y=float(v_hat[1]),
#                           z=float(v_hat[2])))
# 
#     # (3)  How much angular slack (rad) is acceptable?
#     align.threshold_radians = np.deg2rad(20.0)      # ≈ ±20°
# 
#     return req

def wait_until_done(manip_client, cmd_id, timeout=60):
    FeedReq = mp.ManipulationApiFeedbackRequest
    t0 = time.time()

    while time.time() - t0 < timeout:
        fb     = manip_client.manipulation_api_feedback_command(
                     FeedReq(manipulation_cmd_id=cmd_id))
        state  = fb.current_state

        # Helpful for debugging – prints the human-readable enum name.
        print(mp.ManipulationFeedbackState.Name(state))

        if state == mp.MANIP_STATE_DONE:
            return True

        # Treat only true failure / stall conditions as errors.
        if state in (mp.MANIP_STATE_GRASP_FAILED,
                     mp.MANIP_STATE_PLACE_FAILED,
                     ):
            return False

        time.sleep(0.1)

    return False   # timed out
