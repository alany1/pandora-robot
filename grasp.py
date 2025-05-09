# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to use Spot's arm.
"""

import argparse
import sys
import time

import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import numpy as np
from bosdyn.api import arm_command_pb2, geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         blocking_stand)
from bosdyn.client.robot_state import RobotStateClient

from utils.manipulation import (
    wait_until_done,
    make_walk_request,
)
from utils.pose_streamer import FiducialPoseStreamerSE3
from navigate import _wait_for_traj_stop
def quat_from_two_vectors(a, b):
    """Return quaternion q = [w, x, y, z] so that q ⊗ a ⊗ q⁻¹ = b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = np.dot(a, b)

    if dot > 0.999999:                 # already aligned
        return np.array([1.0, 0.0, 0.0, 0.0])

    if dot < -0.999999:                # opposite directions → 180° turn
        # pick an arbitrary orthogonal axis
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(a, axis)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])  # w = 0  ⇒  π rad rotation

    cross = np.cross(a, b)
    q = np.array([1.0 + dot, *cross])
    return q / np.linalg.norm(q)

def hand_body_to_arm_cmd(*, point_body, quat_body, robot_state, delta):
    """
    Convert target point and quat in body frame (for the gripper) to RobotCommand.
    delta: how much time in seconds 
    """
    x, y, z = point_body
    hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

    # Rotation as a quaternion
    qw, qx, qy, qz = quat_body
    flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

    flat_body_T_hand = geometry_pb2.SE3Pose(
        position=hand_ewrt_flat_body, rotation=flat_body_Q_hand
    )

    odom_T_flat_body = get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot,
        ODOM_FRAME_NAME,
        BODY_FRAME_NAME,
    )

    odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)

    arm_command = RobotCommandBuilder.arm_pose_command(
        odom_T_hand.x,
        odom_T_hand.y,
        odom_T_hand.z,
        odom_T_hand.rot.w,
        odom_T_hand.rot.x,
        odom_T_hand.rot.y,
        odom_T_hand.rot.z,
        ODOM_FRAME_NAME,
        delta,
    )
    
    return arm_command

def get_translation_command(*, robot_state, dx, dy, dheading):
    """
    the dx, dy, dheading are in the BODY frame
    """
    snapshot = robot_state.kinematic_state.transforms_snapshot
    traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
        goal_x_rt_body=dx,
        goal_y_rt_body=dy,
        goal_heading_rt_body=dheading,
        frame_tree_snapshot=snapshot,  # not needed for body-frame
    )
    return traj_cmd

def hello_arm(config):
    """A simple example of using the Boston Dynamics API to command Spot's arm."""

    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('HelloSpotClient')
    robot = sdk.create_robot("192.168.80.3")
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    manip_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=False, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')

        streamer = FiducialPoseStreamerSE3(robot_ip="192.168.80.3", tag_id=16)
        tag_aria_tfs = "/Users/alanyu/Library/CloudStorage/Dropbox/spark_aria/real/spot_room_v1/vol_fusion_v1/sg_obs/world_tfs.npy"
        tag_aria_tfs = np.load(tag_aria_tfs)
        tag_aria_tf = tag_aria_tfs[0]
        ALIGN_PUPIL_TO_BD = np.array(
            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float
        )
        
        # standoff point
        # target_point = [-0.589,  0.064, -0.134]
        target_point = [-0.755, 0.181, -0.205]
        grasp_point = [-0.8074752,  0.18065326, -0.50076877]
        
        target_point = [-0.97574752, 0.18065326, -0.50576877]
        grasp_point = [-1.0574752, 0.19065326, -0.50576877]
        for _ in range(20):
            robot_tag_tf = streamer.body_transform_matrix()
        
        robot_aria_tf = tag_aria_tf @ ALIGN_PUPIL_TO_BD @ robot_tag_tf
        standoff_to_robot = (np.linalg.inv(robot_aria_tf) @ np.array([[*target_point, 1]]).T).T[0]
        grasp_to_robot = (np.linalg.inv(robot_aria_tf) @ np.array([[*grasp_point, 1]]).T).T[0]
        
        x, y, z = [standoff_to_robot[0], standoff_to_robot[1], -standoff_to_robot[2]]
        x2, y2, z2 = [grasp_to_robot[0], grasp_to_robot[1], -grasp_to_robot[2]]

        p_standoff = np.array([x, y, z])  # current hand position (m)
        p_grasp = np.array([x2, y2, z2])  # target grasp point   (m)

        v = p_grasp - p_standoff  # 3-D vector from standoff → grasp
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            raise ValueError("Standoff and grasp are basically the same point.")
        v_hat = v / v_norm  # unit direction vector

        hand_x = np.array([1.0, 0.0, 0.0])  # +X of the hand/tool frame
        q_align = quat_from_two_vectors(hand_x, v_hat)
        
        # robot_state = robot_state_client.get_robot_state()
        # backup_cmd = get_translation_command(robot_state=robot_state, dx=0.0, dy=0.0, dheading=0,)
        # cmd_id = command_client.robot_command(backup_cmd, end_time_secs=time.time() + 5.0)
        # if not _wait_for_traj_stop(command_client, cmd_id):
        #     print("⚠️  Spot may not have reached the full 0.5 m, continuing anyway.")
        
        #################################
        robot_state = robot_state_client.get_robot_state()
        odom_T_hand = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            ODOM_FRAME_NAME,
            BODY_FRAME_NAME,
        ) * math_helpers.SE3Pose(
            x=p_standoff[0],
            y=p_standoff[1],
            z=p_standoff[2],
            rot=math_helpers.Quat(*q_align),
        )
        
        walk_req = make_walk_request(odom_T_hand, standoff=0.4)
        walk_resp = manip_client.manipulation_api_command(walk_req)

        if not wait_until_done(manip_client, walk_resp.manipulation_cmd_id):
            print("Walk step failed — aborting.")
            return

        robot_state = robot_state_client.get_robot_state()
        # arm_command = hand_body_to_arm_cmd(point_body=p_standoff,
        #                      quat_body=q_align,
        #                      robot_state=robot_state,
        #                      delta=5.0)
        

        arm_command = RobotCommandBuilder.arm_pose_command(odom_T_hand.x, 
                                                           odom_T_hand.y, 
                                                           odom_T_hand.z,
                                                           odom_T_hand.rot.w,
                                                           odom_T_hand.rot.x,
                                                           odom_T_hand.rot.y,
                                                           odom_T_hand.rot.z,
                                                           ODOM_FRAME_NAME,
                                                           5.0,)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.4)

        # Combine the arm and gripper commands into one RobotCommand
        command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)
        # 
        # Send the request
        cmd_id = command_client.robot_command(command)
        robot.logger.info('Moving arm to position 1.')
        if not _wait_for_traj_stop(command_client, cmd_id):
            robot.logger.info("⚠️  Spot may not have reached the full 0.5 m, continuing anyway.")
        
        time.sleep(2)
        return
        ################3
        robot_tag_tf = streamer.body_transform_matrix()

        robot_aria_tf = tag_aria_tf @ ALIGN_PUPIL_TO_BD @ robot_tag_tf
        grasp_to_robot = (
            np.linalg.inv(robot_aria_tf) @ np.array([[*grasp_point, 1]]).T
        ).T[0]
        x2, y2, z2 = [grasp_to_robot[0], grasp_to_robot[1], -grasp_to_robot[2]]

        p_grasp = np.array([x2, y2, z2])  # target grasp point   (m)
        body_T_hand = get_a_tform_b(
            robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
            BODY_FRAME_NAME,  # or GRAV_ALIGNED_BODY_FRAME_NAME
            HAND_FRAME_NAME,
        )  # literal "hand" works too

        current_hand_pos = np.array([body_T_hand.position.x, 
                                     body_T_hand.position.y, 
                                     body_T_hand.position.z])

        print("current hand pos", current_hand_pos)
        v_new = p_grasp - current_hand_pos  # 3-D vector from standoff → grasp
        v_norm_new = np.linalg.norm(v_new)
        if v_norm_new < 1e-6:
            raise ValueError("Standoff and grasp are basically the same point.")
        v_hat_new = v_new / v_norm_new  # unit direction vector
        q_align_new = quat_from_two_vectors(hand_x, v_hat_new)
        
        robot_state = robot_state_client.get_robot_state()
        odom_T_hand = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            ODOM_FRAME_NAME,
            BODY_FRAME_NAME,
        ) * math_helpers.SE3Pose(
            x=p_grasp[0],
            y=p_grasp[1],
            z=p_grasp[2],
            rot=math_helpers.Quat(*q_align_new),
        )

        walk_req = make_walk_request(odom_T_hand, standoff=0.4)
        walk_resp = manip_client.manipulation_api_command(walk_req)
        if not wait_until_done(manip_client, walk_resp.manipulation_cmd_id):
            print("Walk step failed — aborting.")
            return

        time.sleep(2.0)

        # 3. Use it, but navigate first.
        # grasp_command = hand_body_to_arm_cmd(point_body=p_standoff + 1.0*v,
        #                                      quat_body=q_align,
        #                                      robot_state=robot_state,
        #                                      delta=5.0)

        grasp_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x,
            odom_T_hand.y,
            odom_T_hand.z,
            odom_T_hand.rot.w,
            odom_T_hand.rot.x,
            odom_T_hand.rot.y,
            odom_T_hand.rot.z,
            ODOM_FRAME_NAME,
            5.0,
        )


        cmd_id = command_client.robot_command(grasp_command, end_time_secs=time.time()+5.0)
        robot.logger.info('Moving arm to position 2.')
        if not _wait_for_traj_stop(command_client, cmd_id):
            robot.logger.info("⚠️  Spot may not have reached the full 0.5 m, continuing anyway.")
        # block_until_arm_arrives_with_prints(robot, command_client, cmd_id)
        # #
        time.sleep(2.0)
        
        # close gripper
        close_gripper_command = RobotCommandBuilder.claw_gripper_close_command(0.0, max_torque=5)
        cmd_id = command_client.robot_command(close_gripper_command, end_time_secs=time.time()+2.0)
        robot.logger.info('Closing gripper.')

        time.sleep(1.0)


        robot_state = robot_state_client.get_robot_state()
        robot_tag_tf = streamer.body_transform_matrix()

        robot_aria_tf = tag_aria_tf @ ALIGN_PUPIL_TO_BD @ robot_tag_tf
        grasp_to_robot = (
                np.linalg.inv(robot_aria_tf) @ np.array([[*target_point, 1]]).T
        ).T[0]
        p_standoff = np.array([grasp_to_robot[0], grasp_to_robot[1], -grasp_to_robot[2]])

        grasp_command = hand_body_to_arm_cmd(point_body=p_standoff,
                                             quat_body=q_align_new,
                                             robot_state=robot_state,
                                             delta=2.0)
        
        cmd_id = command_client.robot_command(grasp_command, end_time_secs=time.time()+5.0)
        robot.logger.info('Moving arm back to standoff.')
        if not _wait_for_traj_stop(command_client, cmd_id):
            robot.logger.info("⚠️  Spot may not have reached the full 0.5 m, continuing anyway.")
        # 
        # block_until_arm_arrives_with_prints(robot, command_client, cmd_id)
        # 
        time.sleep(5)

        robot.logger.info('Done.')

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), 'Robot power off failed.'
        robot.logger.info('Robot safely powered off.')


def block_until_arm_arrives_with_prints(robot, command_client, cmd_id):
    """Block until the arm arrives at the goal and print the distance remaining.
        Note: a version of this function is available as a helper in robot_command
        without the prints.
    """
    while True:
        feedback_resp = command_client.robot_command_feedback(cmd_id)
        measured_pos_distance_to_goal = feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.measured_pos_distance_to_goal
        measured_rot_distance_to_goal = feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.measured_rot_distance_to_goal
        robot.logger.info('Distance to go: %.2f meters, %.2f radians',
                          measured_pos_distance_to_goal, measured_rot_distance_to_goal)

        if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
            robot.logger.info('Move complete.')
            break
        time.sleep(0.1)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args()
    try:
        hello_arm(options)
        return True
    except Exception:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)
