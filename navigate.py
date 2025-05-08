#!/usr/bin/env python3
"""
Half-meter forward step for Boston Dynamics Spot.

Usage:
    python move_half_meter.py --robot-ip 10.31.169.114 --username admin --password "pwd"
"""
import argparse
import sys
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.api import basic_command_pb2, geometry_pb2
from bosdyn.client.robot_state import RobotStateClient


# -----------------------------------------------------------------------------#
# Helper feedback-blocking utilities (copied in spirit from your script)        #
# -----------------------------------------------------------------------------#
def _wait_for_stand(cmd_client, cmd_id, timeout=10.0):
    """Return True once Stand feedback reports STATUS_IS_STANDING."""
    start = time.time()
    while time.time() - start < timeout:
        fb = cmd_client.robot_command_feedback(cmd_id)
        status = (
            fb.feedback.synchronized_feedback.mobility_command_feedback
            .stand_feedback.status
        )
        if status == basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING:  #:contentReference[oaicite:0]{index=0}
            return True
        time.sleep(0.1)
    return False


def _wait_for_sit(cmd_client, cmd_id, timeout=10.0):
    """Return True once Sit feedback reports STATUS_IS_SITTING."""
    start = time.time()
    while time.time() - start < timeout:
        fb = cmd_client.robot_command_feedback(cmd_id)
        status = (
            fb.feedback.synchronized_feedback.mobility_command_feedback
            .sit_feedback.status
        )
        if status == basic_command_pb2.SitCommand.Feedback.STATUS_IS_SITTING:  #:contentReference[oaicite:1]{index=1}
            return True
        time.sleep(0.1)
    return False


def _wait_for_traj_stop(cmd_client, cmd_id, timeout=10.0):
    """Block until SE2Trajectory reports STATUS_STOPPED."""
    start = time.time()
    while time.time() - start < timeout:
        fb = cmd_client.robot_command_feedback(cmd_id)
        status = (
            fb.feedback.synchronized_feedback.mobility_command_feedback
            .se2_trajectory_feedback.status
        )
        if status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_STOPPED:
            return True
        time.sleep(0.1)
    return False


# -----------------------------------------------------------------------------#
# Main routine                                                                  #
# -----------------------------------------------------------------------------#
def drive_half_meter(robot):
    """Power, stand, walk 0.5 m forward, sit, power-off."""
    cmd_client: RobotCommandClient = robot.ensure_client(
        RobotCommandClient.default_service_name
    )
    state_client: RobotStateClient = robot.ensure_client(
        RobotStateClient.default_service_name
    )
    # Take a lease for the entire session.
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    lease = lease_client.take("body")
    with LeaseKeepAlive(lease_client, must_acquire=False):

        # Power on & stand -----------------------------------------------------
        robot.power_on(timeout_sec=20)
        stand_cmd = RobotCommandBuilder.synchro_stand_command()
        stand_id = cmd_client.robot_command(stand_cmd)
        if not _wait_for_stand(cmd_client, stand_id):
            raise RuntimeError("Stand command timed out")

        # Build BODY-frame 0.5 m SE2 trajectory -------------------------------
        snapshot = state_client.get_robot_state().kinematic_state.transforms_snapshot
        traj_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
            goal_x_rt_body=0,  # forward half-meter
            goal_y_rt_body=-0.22,
            goal_heading_rt_body=0,
            frame_tree_snapshot=snapshot,  # not needed for body-frame
        )
        traj_id = cmd_client.robot_command(
            traj_cmd, end_time_secs=time.time() + 5.0  # generous 5 s window
        )
        if not _wait_for_traj_stop(cmd_client, traj_id):
            print("⚠️  Spot may not have reached the full 0.5 m, continuing anyway.")

        # Sit & power off ------------------------------------------------------
        sit_cmd = RobotCommandBuilder.synchro_sit_command()
        sit_id = cmd_client.robot_command(sit_cmd)
        _wait_for_sit(cmd_client, sit_id)  # best-effort; ignore result
        robot.power_off(cut_immediately=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="192.168.80.3", required=False, help="Robot IPv4, e.g. 10.31.169.114")
    # ap.add_argument("--username", default="admin")
    # ap.add_argument("--password", required=True)
    args = ap.parse_args()

    # SDK setup exactly like your initialize_robot() --------------------------
    sdk = bosdyn.client.create_standard_sdk("HalfMeterClient")
    robot = sdk.create_robot(args.robot_ip)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    try:
        drive_half_meter(robot)
        print("Finished: Spot is sitting and powered off.")
    except Exception as err:
        print(f"Exception: {err}", file=sys.stderr)
        robot.power_off(cut_immediately=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
