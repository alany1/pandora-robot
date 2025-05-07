import cv2
import numpy as np
import logging
import time
import pickle

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient

import apriltag

LOGGER = logging.getLogger()

from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from google.protobuf.json_format import MessageToDict
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, ODOM_FRAME_NAME,GRAV_ALIGNED_BODY_FRAME_NAME,
                                         get_se2_a_tform_b, get_a_tform_b)
from bosdyn.client import math_helpers
from bosdyn.api import world_object_pb2

def se2_pose_to_np(pose):
    return np.array([pose.x, pose.y, pose.angle])

APRIL_TAG_SIZE = 0.045 # meters
CHAIR_BACK_TAG_NUMBER = 6
CHAIR_CENTER_TAG_NUMBER = 8
TRAJECTORY_FRAME_RATE = 1/5
APRIL_TAG_NUMBERS = [31]
APRIL_TAG_NUMBERS = [16]
IMAGE_SOURCES = [ 'frontleft_fisheye_image',
               'frontright_fisheye_image',
               ]
DEPTH_IMAGE_SOURCES = [ 'frontleft_fisheye_image',
               'frontleft_depth_in_visual_frame',
               'frontright_fisheye_image',
               'frontright_depth_in_visual_frame',
               ]
HAND_DEPTH_IMAGE_SOURCES = ['hand_depth_in_hand_color_frame',]
# HAND_DEPTH_IMAGE_SOURCES = ['frontright_depth_in_visual_frame',]

FID_TFORMs_FID1 = [math_helpers.SE2Pose(x=0, y=0, angle=0)]
REFERENCE_TIME= time.time()

class SynchronousRecordRobotPositionAndState():
    def __init__(self, robot_state_client, world_object_client, online_mode=False):
        self.step_counter = 0

        self.joint_states = []
        self.body_transforms_in_fid_frame = []
        self.body_velocities_in_body_frame = []
        self.record_times = []

        self.world_object_client = world_object_client
        self.robot_state_client = robot_state_client

        self.request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        self.fid_tforms_odom = self.localize_odom()

        self.transforms = []

        self.results = []
        self.online_mode = online_mode

    def localize_body(self, transforms):
        fiducial_objects = self.world_object_client.list_world_objects(object_type=self.request_fiducials).world_objects
        fiducials = [None] * len(APRIL_TAG_NUMBERS)
        for f in fiducial_objects:
            for i in range(len(APRIL_TAG_NUMBERS)):
                if APRIL_TAG_NUMBERS[i] == f.apriltag_properties.tag_id:
                    fiducials[i] = f
        if None in fiducials:
            # print(fiducials)
            raise Exception(f'april tag {APRIL_TAG_NUMBERS[0]} not found')
        # print(f.transforms_snapshot)
        body_tforms_fid = \
            [get_se2_a_tform_b(transforms, GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME) * get_se2_a_tform_b(
                f.transforms_snapshot, ODOM_FRAME_NAME, f.apriltag_properties.frame_name_fiducial_filtered) for f in
            fiducials]
        for i in range(len(body_tforms_fid)):
            body_tforms_fid[i] = (body_tforms_fid[i] * FID_TFORMs_FID1[i]).inverse()
            pass
        fid1_tforms_body = [se2_pose_to_np(t) for t in body_tforms_fid]
        fid1_tforms_body = np.mean(np.array(fid1_tforms_body), axis=0)
        return math_helpers.SE2Pose(x=fid1_tforms_body[0], y=fid1_tforms_body[1], angle=fid1_tforms_body[2])

    def localize_odom(self):
        fiducial_objects = self.world_object_client.list_world_objects(object_type=self.request_fiducials).world_objects
        fiducials = [None] * len(APRIL_TAG_NUMBERS)
        for f in fiducial_objects:
            for i in range(len(APRIL_TAG_NUMBERS)):
                if APRIL_TAG_NUMBERS[i] == f.apriltag_properties.tag_id:
                    fiducials[i] = f
        # print(fiducials)
        if None in fiducials:
            print(fiducials)
            raise Exception(f'april tag {APRIL_TAG_NUMBERS[0]} not found')
        # print(f.transforms_snapshot)
        odom_tforms_fid = [get_se2_a_tform_b(f.transforms_snapshot, ODOM_FRAME_NAME,
                                             f.apriltag_properties.frame_name_fiducial_filtered) for f in fiducials]
        for i in range(len(odom_tforms_fid)):
            odom_tforms_fid[i] = (odom_tforms_fid[i] * FID_TFORMs_FID1[i]).inverse()
            pass
        fid1_tforms_odom = [se2_pose_to_np(t) for t in odom_tforms_fid]
        fid1_tforms_odom = np.mean(np.array(fid1_tforms_odom), axis=0)
        return math_helpers.SE2Pose(x=fid1_tforms_odom[0], y=fid1_tforms_odom[1], angle=fid1_tforms_odom[2])

    def update(self):
        result = self.robot_state_client.get_robot_state()
        self.results.append(result)
        self.record_times.append(time.time() - REFERENCE_TIME)
        self.step_counter += 1
        if self.online_mode: self.process_latest_result()

    def process_results(self):
        for result in self.results:
            kinematic_state = MessageToDict(result.kinematic_state)
            transforms = result.kinematic_state.transforms_snapshot

            current_joint_states = np.zeros((20, 3))
            for i, joint in enumerate(kinematic_state['jointStates']):
                current_joint_states[i][0] = joint['position']
                current_joint_states[i][1] = joint['velocity']
                current_joint_states[i][2] = joint['load']

            odom_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

            xvel = kinematic_state['velocityOfBodyInOdom']['linear']['x']
            yvel = kinematic_state['velocityOfBodyInOdom']['linear']['y']
            rvel = kinematic_state['velocityOfBodyInOdom']['angular']['z']
            velocity_in_odom_frame = math_helpers.SE2Velocity(x=xvel, y=yvel, angular=rvel)
            velocity_in_body_frame = math_helpers.SE2Velocity.from_vector(
                odom_tform_body.inverse().to_adjoint_matrix() @ velocity_in_odom_frame.to_vector())
            velocity_in_body_frame = velocity_in_body_frame.to_vector().flatten()

            fid_tform_body = self.fid_tforms_odom * odom_tform_body

            body_pos_in_fid_frame = se2_pose_to_np(fid_tform_body)

            self.joint_states.append(current_joint_states)
            self.body_transforms_in_fid_frame.append(body_pos_in_fid_frame)
            self.body_velocities_in_body_frame.append(velocity_in_body_frame)
    def process_latest_result(self):
        result = self.results[-1]
        kinematic_state = MessageToDict(result.kinematic_state)
        transforms = result.kinematic_state.transforms_snapshot
        self.transforms.append((transforms))

        current_joint_states = np.zeros((20, 3))
        for i, joint in enumerate(kinematic_state['jointStates']):
            current_joint_states[i][0] = joint['position']
            current_joint_states[i][1] = joint['velocity']
            current_joint_states[i][2] = joint['load']

        odom_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        xvel = kinematic_state['velocityOfBodyInOdom']['linear']['x']
        yvel = kinematic_state['velocityOfBodyInOdom']['linear']['y']
        rvel = kinematic_state['velocityOfBodyInOdom']['angular']['z']
        velocity_in_odom_frame = math_helpers.SE2Velocity(x=xvel, y=yvel, angular=rvel)
        velocity_in_body_frame = math_helpers.SE2Velocity.from_vector(
            odom_tform_body.inverse().to_adjoint_matrix() @ velocity_in_odom_frame.to_vector())
        velocity_in_body_frame = velocity_in_body_frame.to_vector().flatten()

        fid_tform_body = self.fid_tforms_odom * odom_tform_body

        body_pos_in_fid_frame = se2_pose_to_np(fid_tform_body)

        self.joint_states.append(current_joint_states)
        self.body_transforms_in_fid_frame.append(body_pos_in_fid_frame)
        self.body_velocities_in_body_frame.append(velocity_in_body_frame)

    def save_trajectory(self, trajectory_file):
        self.process_results()
        traj = {'record_times': np.array(self.record_times),
                'joint_states': np.array(self.joint_states),
                'body_pos_in_fid': np.array(self.body_transforms_in_fid_frame),
                'body_vel_in_body': np.array(self.body_velocities_in_body_frame)}

        with open(trajectory_file, 'wb') as f:
            pickle.dump(traj, f)


class SynchronousRecordChairImageAndPose():

    def __init__(self, image_client, image_sources, robot_state_task, online_mode=False):
        self.step_counter = 0

        self.chair_images = []
        self.record_times = []

        self.image_sources = image_sources
        self.image_client = image_client

        self.robot_state_task = robot_state_task
        self.chair_states_in_body_frame = []

        options = apriltag.DetectorOptions(families="tag36h11")
        self.detector = apriltag.Detector(options)

        self.results = []
        self.point_clouds = []
        self.online_mode = online_mode

    def update(self):
        result = self.image_client.get_image_from_sources(self.image_sources)
        self.results.append(result)
        self.record_times.append(time.time() - REFERENCE_TIME)
        self.step_counter += 1
        if self.online_mode: self.process_latest_result()




    def process_results(self):
        for t, result in enumerate(self.results):
            final_im = []

            for i, source_name in enumerate(self.image_sources):
                image_response = result[i]
                image_capture, image_source = image_response.shot, image_response.source
                focal_length, principal_point = image_source.pinhole.intrinsics.focal_length, image_source.pinhole.intrinsics.principal_point
                camera_params = [focal_length.x, focal_length.y, principal_point.x, principal_point.y]

                # print(image_capture.transforms_snapshot, image_capture.frame_name_image_sensor)
                if 'depth' in source_name:
                    im = np.frombuffer(image_capture.image.data, dtype=np.uint16)
                    im = im.reshape(image_capture.image.rows,
                                    image_capture.image.cols, 1)
                    im = ((im - np.min(im)) / (np.max(im) - np.min(im)) * 255).astype(np.uint8)
                else:
                    im = cv2.imdecode(np.frombuffer(image_capture.image.data, dtype=np.uint8), -1)
                    im = im if len(im.shape) == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

                chair_state = [None] * 2

                if 'depth' not in source_name:
                    detections, dimg = self.detector.detect(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), return_image=True)
                    im = im // 2 + dimg[:, :, None] // 2
                    body_tform_cam = get_a_tform_b(self.robot_state_task.results[t].kinematic_state.transforms_snapshot,
                                                   GRAV_ALIGNED_BODY_FRAME_NAME,
                                                   BODY_FRAME_NAME) * get_a_tform_b(image_capture.transforms_snapshot,
                                                                                    BODY_FRAME_NAME,
                                                                                    image_capture.frame_name_image_sensor)
                    for d in detections:
                        if d.center[0] < 240:
                            chair_back = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                            cam_tform_chair_back = math_helpers.SE3Pose.from_matrix(chair_back)
                            body_tform_chair_back = body_tform_cam * cam_tform_chair_back
                            body_tform_chair_back = body_tform_chair_back.get_closest_se2_transform()
                            if chair_state[0] is None:
                                chair_state[0] = se2_pose_to_np(body_tform_chair_back)
                            else:
                                chair_state[0] += se2_pose_to_np(body_tform_chair_back)
                                chair_state[0] /= 2

                        if d.center[0] > 240:
                            chair_leg = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                            cam_tform_chair_leg = math_helpers.SE3Pose.from_matrix(chair_leg)
                            body_tform_chair_leg = body_tform_cam * cam_tform_chair_leg
                            body_tform_chair_leg = body_tform_chair_leg.get_closest_se2_transform()
                            if 'frontright' in source_name: chair_state[1] = se2_pose_to_np(body_tform_chair_leg)

                    self.chair_states_in_body_frame.append(chair_state)
                    pass
                final_im.append(im)

            img = np.concatenate(final_im, axis=-1)
            self.chair_images.append(img)

    def process_latest_result(self):
        result = self.results[-1]
        final_im = []
        point_clouds = []

        for i, source_name in enumerate(self.image_sources):
            image_response = result[i]
            image_capture, image_source = image_response.shot, image_response.source
            focal_length, principal_point = image_source.pinhole.intrinsics.focal_length, image_source.pinhole.intrinsics.principal_point
            camera_params = [focal_length.x, focal_length.y, principal_point.x, principal_point.y]

            # print(image_capture.transforms_snapshot, image_capture.frame_name_image_sensor)
            if 'depth' in source_name:
                point_cloud = bosdyn.client.image.depth_image_to_pointcloud(image_response)
                camera_coordinates = get_a_tform_b(image_capture.transforms_snapshot, BODY_FRAME_NAME,
                                                   image_capture.frame_name_image_sensor)
                point_clouds.append(camera_coordinates.transform_cloud(point_cloud))

                im = np.frombuffer(image_capture.image.data, dtype=np.uint16)
                im = im.reshape(image_capture.image.rows,
                                image_capture.image.cols, 1)
                im = ((im - np.min(im)) / (np.max(im) - np.min(im)) * 255).astype(np.uint8)
            else:
                im = cv2.imdecode(np.frombuffer(image_capture.image.data, dtype=np.uint8), -1)
                im = im if len(im.shape) == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

            chair_state = [None] * 2

            if 'depth' not in source_name:
                detections, dimg = self.detector.detect(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), return_image=True)
                im = im // 2 + dimg[:, :, None] // 2
                body_tform_cam = get_a_tform_b(self.robot_state_task.results[-1].kinematic_state.transforms_snapshot,
                                               GRAV_ALIGNED_BODY_FRAME_NAME,
                                               BODY_FRAME_NAME) * get_a_tform_b(image_capture.transforms_snapshot,
                                                                                BODY_FRAME_NAME,
                                                                                image_capture.frame_name_image_sensor)
                for d in detections:
                    if d.center[0] < 240:
                        chair_back = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                        cam_tform_chair_back = math_helpers.SE3Pose.from_matrix(chair_back)
                        body_tform_chair_back = body_tform_cam * cam_tform_chair_back
                        body_tform_chair_back = body_tform_chair_back.get_closest_se2_transform()
                        if chair_state is None:
                            chair_state[0] = se2_pose_to_np(body_tform_chair_back)
                        else:
                            chair_state[0] += se2_pose_to_np(body_tform_chair_back)
                            chair_state[0] /= 2

                    if d.center[0] > 240:
                        chair_leg = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                        cam_tform_chair_leg = math_helpers.SE3Pose.from_matrix(chair_leg)
                        body_tform_chair_leg = body_tform_cam * cam_tform_chair_leg
                        body_tform_chair_leg = body_tform_chair_leg.get_closest_se2_transform()
                        if 'frontright' in source_name: chair_state[1] = se2_pose_to_np(body_tform_chair_leg)

                if chair_state[0] is None and len(self.chair_states_in_body_frame) > 0:
                    self.chair_states_in_body_frame.append(self.chair_states_in_body_frame[-1])
                else:
                    self.chair_states_in_body_frame.append(chair_state)
                pass
            final_im.append(im)

        img = np.concatenate(final_im, axis=-1)
        self.chair_images.append(img)

        if point_clouds: point_clouds = np.concatenate(point_clouds, axis=0)
        self.point_clouds.append(point_clouds)

    def save_trajectory(self, image_file):
        self.process_results()
        traj = {'record_times': np.array(self.record_times),
                'chair_images': np.array(self.chair_images),
                'chair_states_in_body_frame': self.chair_states_in_body_frame}
        with open(image_file, 'wb') as f:
            pickle.dump(traj, f)


class OldSynchronousRecordChairImageAndPose():

    def __init__(self, image_client, image_sources, robot_state_task):
        self.step_counter = 0

        self.chair_images = []
        self.record_times = []

        self.image_sources = image_sources
        self.image_client = image_client

        self.robot_state_task = robot_state_task
        self.chair_states_in_body_frame = []

        options = apriltag.DetectorOptions(families="tag36h11")
        self.detector = apriltag.Detector(options)

        self.results = []
        self.point_clouds = []

    def update(self):
        result = self.image_client.get_image_from_sources(self.image_sources)
        self.results.append(result)
        self.record_times.append(time.time() - REFERENCE_TIME)
        self.step_counter += 1




    def process_results(self):
        for t, result in enumerate(self.results):
            final_im = []

            for i, source_name in enumerate(self.image_sources):
                image_response = result[i]
                image_capture, image_source = image_response.shot, image_response.source
                focal_length, principal_point = image_source.pinhole.intrinsics.focal_length, image_source.pinhole.intrinsics.principal_point
                camera_params = [focal_length.x, focal_length.y, principal_point.x, principal_point.y]

                # print(image_capture.transforms_snapshot, image_capture.frame_name_image_sensor)
                if 'depth' in source_name:
                    im = np.frombuffer(image_capture.image.data, dtype=np.uint16)
                    im = im.reshape(image_capture.image.rows,
                                    image_capture.image.cols, 1)
                    im = ((im - np.min(im)) / (np.max(im) - np.min(im)) * 255).astype(np.uint8)
                else:
                    im = cv2.imdecode(np.frombuffer(image_capture.image.data, dtype=np.uint8), -1)
                    im = im if len(im.shape) == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

                chair_state = [None] * 2

                chair_center_location = None
                chair_center_rotation = None

                if 'depth' not in source_name:
                    detections, dimg = self.detector.detect(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), return_image=True)
                    im = im // 2 + dimg[:, :, None] // 2
                    body_tform_cam = get_a_tform_b(self.robot_state_task.results[t].kinematic_state.transforms_snapshot,
                                                   GRAV_ALIGNED_BODY_FRAME_NAME,
                                                   BODY_FRAME_NAME) * get_a_tform_b(image_capture.transforms_snapshot,
                                                                                    BODY_FRAME_NAME,
                                                                                    image_capture.frame_name_image_sensor)
                    for d in detections:
                        if d.center[0] < 240:
                            chair_back = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                            cam_tform_chair_back = math_helpers.SE3Pose.from_matrix(chair_back)
                            body_tform_chair_back = body_tform_cam * cam_tform_chair_back
                            body_tform_chair_back = body_tform_chair_back.get_closest_se2_transform()
                            if chair_center_rotation is not None:
                                chair_center_rotation += se2_pose_to_np(body_tform_chair_back)[2]
                                chair_center_rotation = chair_center_rotation / 2
                            else:
                                chair_center_rotation = se2_pose_to_np(body_tform_chair_back)[2]

                        if d.center[0] > 240:
                            chair_center = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                            cam_tform_chair_center = math_helpers.SE3Pose.from_matrix(chair_center)
                            body_tform_chair_center = body_tform_cam * cam_tform_chair_center
                            body_tform_chair_center = body_tform_chair_center.get_closest_se2_transform()
                            if chair_center_location[0] is not None:
                                chair_center_location += se2_pose_to_np(body_tform_chair_center)[:2]
                                chair_center_location /= 2
                            else:
                                chair_center_location = se2_pose_to_np(body_tform_chair_center)[:2]

                    if chair_center_location is not None and chair_center_rotation is not None:
                        chair_state[0] = np.array([*list(chair_center_rotation), chair_center_location])

                    self.chair_states_in_body_frame.append(chair_state)
                    pass
                final_im.append(im)

            img = np.concatenate(final_im, axis=-1)
            self.chair_images.append(img)

    def process_latest_result(self):
        result = self.results[-1]
        final_im = []
        point_clouds = []

        for i, source_name in enumerate(self.image_sources):
            image_response = result[i]
            image_capture, image_source = image_response.shot, image_response.source
            focal_length, principal_point = image_source.pinhole.intrinsics.focal_length, image_source.pinhole.intrinsics.principal_point
            camera_params = [focal_length.x, focal_length.y, principal_point.x, principal_point.y]

            # print(image_capture.transforms_snapshot, image_capture.frame_name_image_sensor)
            if 'depth' in source_name:
                point_cloud = bosdyn.client.image.depth_image_to_pointcloud(image_response)
                camera_coordinates = get_a_tform_b(image_capture.transforms_snapshot, BODY_FRAME_NAME,
                                                   image_capture.frame_name_image_sensor)
                point_clouds.append(camera_coordinates.transform_cloud(point_cloud))

                im = np.frombuffer(image_capture.image.data, dtype=np.uint16)
                im = im.reshape(image_capture.image.rows,
                                image_capture.image.cols, 1)
                im = ((im - np.min(im)) / (np.max(im) - np.min(im)) * 255).astype(np.uint8)
            else:
                im = cv2.imdecode(np.frombuffer(image_capture.image.data, dtype=np.uint8), -1)
                im = im if len(im.shape) == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

            chair_state = [None] * 2

            chair_center_location = None
            chair_center_rotation = None

            if 'depth' not in source_name:
                detections, dimg = self.detector.detect(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), return_image=True)
                im = im // 2 + dimg[:, :, None] // 2
                body_tform_cam = get_a_tform_b(self.robot_state_task.results[-1].kinematic_state.transforms_snapshot,
                                               GRAV_ALIGNED_BODY_FRAME_NAME,
                                               BODY_FRAME_NAME) * get_a_tform_b(image_capture.transforms_snapshot,
                                                                                BODY_FRAME_NAME,
                                                                                image_capture.frame_name_image_sensor)
                for d in detections:
                    if d.center[0] < 240:
                        chair_back = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                        cam_tform_chair_back = math_helpers.SE3Pose.from_matrix(chair_back)
                        body_tform_chair_back = body_tform_cam * cam_tform_chair_back
                        body_tform_chair_back = body_tform_chair_back.get_closest_se2_transform()
                        if chair_center_rotation is not None:
                            chair_center_rotation += se2_pose_to_np(body_tform_chair_back)[2]
                            chair_center_rotation = chair_center_rotation / 2
                        else:
                            chair_center_rotation = se2_pose_to_np(body_tform_chair_back)[2]

                    if d.center[0] > 240:
                        chair_center = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                        cam_tform_chair_center = math_helpers.SE3Pose.from_matrix(chair_center)
                        body_tform_chair_center = body_tform_cam * cam_tform_chair_center
                        body_tform_chair_center = body_tform_chair_center.get_closest_se2_transform()
                        if chair_center_location[0] is not None:
                            chair_center_location += se2_pose_to_np(body_tform_chair_center)[:2]
                            chair_center_location /= 2
                        else:
                            chair_center_location = se2_pose_to_np(body_tform_chair_center)[:2]

                if chair_center_location is not None and chair_center_rotation is not None:
                    chair_state[0] = np.array([*list(chair_center_rotation), chair_center_location])

                self.chair_states_in_body_frame.append(chair_state)
                pass
            final_im.append(im)

        img = np.concatenate(final_im, axis=-1)
        self.chair_images.append(img)

        if point_clouds: point_clouds = np.concatenate(point_clouds, axis=0)
        self.point_clouds.append(point_clouds)

    def save_trajectory(self, image_file):
        self.process_results()
        traj = {'record_times': np.array(self.record_times),
                'chair_images': np.array(self.chair_images),
                'chair_states_in_body_frame': self.chair_states_in_body_frame}
        with open(image_file, 'wb') as f:
            pickle.dump(traj, f)


class AsyncRecordRobotPositionAndState(AsyncPeriodicQuery):
    """Grab full state of robot"""

    def __init__(self, robot_state_client, world_object_client, online_mode=False):
        super(AsyncRecordRobotPositionAndState, self).__init__('robot_state', robot_state_client, LOGGER,
                                                               period_sec=TRAJECTORY_FRAME_RATE)
        self.step_counter = 0

        self.joint_states = []
        self.body_transforms_in_fid_frame = []
        self.body_velocities_in_body_frame = []
        self.record_times = []

        self.world_object_client = world_object_client
        self.robot_state_client = robot_state_client

        self.request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        self.odom_tform_fid_3d = None
        self.fid_tforms_odom = self.localize_odom()

        self.results = []

        self.transforms = []
        self.online_mode = online_mode

    def localize_odom(self):
        fiducial_objects = self.world_object_client.list_world_objects(object_type=self.request_fiducials).world_objects
        fiducials = [None] * len(APRIL_TAG_NUMBERS)
        for f in fiducial_objects:
            for i in range(len(APRIL_TAG_NUMBERS)):
                if APRIL_TAG_NUMBERS[i] == f.apriltag_properties.tag_id:
                    fiducials[i] = f
        if None in fiducials:
            print(fiducials)
            raise Exception(f'april tag {APRIL_TAG_NUMBERS[0]} not found')
        odom_tforms_fid = [get_se2_a_tform_b(f.transforms_snapshot, ODOM_FRAME_NAME,f.apriltag_properties.frame_name_fiducial_filtered) for f in fiducials]

        #TODO: average for multiple fid for 3d transform
        odom_tforms_fid_3d = [get_a_tform_b(f.transforms_snapshot, ODOM_FRAME_NAME,f.apriltag_properties.frame_name_fiducial_filtered) for f in fiducials]
        self.odom_tform_fid_3d = odom_tforms_fid_3d[0]

        for i in range(len(odom_tforms_fid)):
            odom_tforms_fid[i] = (odom_tforms_fid[i] * FID_TFORMs_FID1[i]).inverse()
            pass
        fid1_tforms_odom = [se2_pose_to_np(t) for t in odom_tforms_fid]
        fid1_tforms_odom = np.mean(np.array(fid1_tforms_odom), axis=0)
        return math_helpers.SE2Pose(x=fid1_tforms_odom[0], y=fid1_tforms_odom[1], angle=fid1_tforms_odom[2])

    def _start_query(self):
        return self._client.get_robot_state_async()

    def _handle_result(self, result):
        self.results.append(result)
        if MessageToDict(result.behavior_fault_state):
            raise Exception(MessageToDict(result.behavior_fault_state))
        self.record_times.append(time.time() - REFERENCE_TIME)
        self.step_counter += 1
        if self.online_mode: self.process_latest_result()
        super(AsyncRecordRobotPositionAndState, self)._handle_result(result)

    def process_results(self):
        self.joint_states = []
        self.body_transforms_in_fid_frame = []
        self.body_velocities_in_body_frame = []
        for result in self.results:
            kinematic_state = MessageToDict(result.kinematic_state)
            transforms = result.kinematic_state.transforms_snapshot

            current_joint_states = np.zeros((20, 3))
            for i, joint in enumerate(kinematic_state['jointStates']):
                current_joint_states[i][0] = joint['position']
                current_joint_states[i][1] = joint['velocity']
                current_joint_states[i][2] = joint['load']

            odom_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

            xvel = kinematic_state['velocityOfBodyInOdom']['linear']['x']
            yvel = kinematic_state['velocityOfBodyInOdom']['linear']['y']
            rvel = kinematic_state['velocityOfBodyInOdom']['angular']['z']
            velocity_in_odom_frame = math_helpers.SE2Velocity(x=xvel, y=yvel, angular=rvel)
            velocity_in_body_frame = math_helpers.SE2Velocity.from_vector(
                odom_tform_body.inverse().to_adjoint_matrix() @ velocity_in_odom_frame.to_vector())
            velocity_in_body_frame = velocity_in_body_frame.to_vector().flatten()

            fid_tform_body = self.fid_tforms_odom * odom_tform_body

            body_pos_in_fid_frame = se2_pose_to_np(fid_tform_body)

            self.joint_states.append(current_joint_states)
            self.body_transforms_in_fid_frame.append(body_pos_in_fid_frame)
            self.body_velocities_in_body_frame.append(velocity_in_body_frame)

    def process_latest_result(self):
        result = self.results[-1]
        kinematic_state = MessageToDict(result.kinematic_state)
        transforms = result.kinematic_state.transforms_snapshot
        self.transforms.append((transforms))

        current_joint_states = np.zeros((20, 3))
        for i, joint in enumerate(kinematic_state['jointStates']):
            current_joint_states[i][0] = joint['position']
            current_joint_states[i][1] = joint['velocity']
            current_joint_states[i][2] = joint['load']

        odom_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        xvel = kinematic_state['velocityOfBodyInOdom']['linear']['x']
        yvel = kinematic_state['velocityOfBodyInOdom']['linear']['y']
        rvel = kinematic_state['velocityOfBodyInOdom']['angular']['z']
        velocity_in_odom_frame = math_helpers.SE2Velocity(x=xvel, y=yvel, angular=rvel)
        velocity_in_body_frame = math_helpers.SE2Velocity.from_vector(
            odom_tform_body.inverse().to_adjoint_matrix() @ velocity_in_odom_frame.to_vector())
        velocity_in_body_frame = velocity_in_body_frame.to_vector().flatten()

        fid_tform_body = self.fid_tforms_odom * odom_tform_body

        body_pos_in_fid_frame = se2_pose_to_np(fid_tform_body)

        self.joint_states.append(current_joint_states)
        self.body_transforms_in_fid_frame.append(body_pos_in_fid_frame)
        self.body_velocities_in_body_frame.append(velocity_in_body_frame)

    def save_trajectory(self, trajectory_file):
        self.process_results()
        traj = {'record_times': np.array(self.record_times),
                'joint_states': np.array(self.joint_states),
                'body_pos_in_fid': np.array(self.body_transforms_in_fid_frame),
                'body_vel_in_body': np.array(self.body_velocities_in_body_frame)}

        with open(trajectory_file, 'wb') as f:
            pickle.dump(traj, f)


class AsyncRecordChairImageAndPose(AsyncPeriodicQuery):

    def __init__(self, image_client, image_sources, robot_state_task, online_mode = False):
        super(AsyncRecordChairImageAndPose, self).__init__('robot_state', image_client, LOGGER,
                                                           period_sec=TRAJECTORY_FRAME_RATE)
        self.step_counter = 0

        self.chair_images = []
        self.point_clouds_robot_perspective = []
        self.record_times = []
        self.global_point_cloud_fid_frame = []

        self.image_sources = image_sources
        self.image_client = image_client

        self.robot_state_task = robot_state_task
        self.chair_states_in_body_frame = []

        options = apriltag.DetectorOptions(families="tag36h11")
        self.detector = apriltag.Detector(options)

        self.online_mode = online_mode

        self.results = []

    def _start_query(self):
        return self._client.get_image_from_sources_async(self.image_sources)

    def _handle_result(self, result):
        self.results.append(result)
        self.record_times.append(time.time() - REFERENCE_TIME)
        self.step_counter += 1
        if self.online_mode: self.process_latest_result()
        super(AsyncRecordChairImageAndPose, self)._handle_result(result)

    def process_latest_result(self):
        result = self.results[-1]
        final_im = []
        point_clouds_robot_perspective = []

        chair_center_locations = []
        chair_center_rotations = []
        chair_state = [None] * 2

        for i, source_name in enumerate(self.image_sources):
            image_response = result[i]
            image_capture, image_source = image_response.shot, image_response.source
            focal_length, principal_point = image_source.pinhole.intrinsics.focal_length, image_source.pinhole.intrinsics.principal_point
            camera_params = [focal_length.x, focal_length.y, principal_point.x, principal_point.y]

            # print(image_capture.transforms_snapshot, image_capture.frame_name_image_sensor)
            if 'depth' in source_name:
                point_cloud = bosdyn.client.image.depth_image_to_pointcloud(image_response)
                camera_coordinates = get_a_tform_b(image_capture.transforms_snapshot, BODY_FRAME_NAME,
                                                   image_capture.frame_name_image_sensor)
                point_clouds_robot_perspective.append(camera_coordinates.transform_cloud(point_cloud))

                camera_coordinates_odom = get_a_tform_b(image_capture.transforms_snapshot, ODOM_FRAME_NAME,
                                                        image_capture.frame_name_image_sensor)
                camera_coordinates_fid = self.robot_state_task.odom_tform_fid_3d.inverse() * camera_coordinates_odom
                self.global_point_cloud_fid_frame.extend(camera_coordinates_fid.transform_cloud(point_cloud))

                im = np.frombuffer(image_capture.image.data, dtype=np.uint16)
                im = im.reshape(image_capture.image.rows,
                                image_capture.image.cols, 1)
                im = ((im - np.min(im)) / (np.max(im) - np.min(im)) * 255).astype(np.uint8)
            else:
                im = cv2.imdecode(np.frombuffer(image_capture.image.data, dtype=np.uint8), -1)
                im = im if len(im.shape) == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


            if 'depth' not in source_name:
                detections, dimg = self.detector.detect(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), return_image=True)
                im = im // 2 + dimg[:, :, None] // 2
                body_tform_cam = get_a_tform_b(self.robot_state_task.results[-1].kinematic_state.transforms_snapshot,
                                               GRAV_ALIGNED_BODY_FRAME_NAME,
                                               BODY_FRAME_NAME) * get_a_tform_b(image_capture.transforms_snapshot,
                                                                                BODY_FRAME_NAME,
                                                                                image_capture.frame_name_image_sensor)
                for d in detections:
                    if d.tag_id != CHAIR_BACK_TAG_NUMBER:
                        chair_back = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                        cam_tform_chair_back = math_helpers.SE3Pose.from_matrix(chair_back)
                        body_tform_chair_back = body_tform_cam * cam_tform_chair_back
                        body_tform_chair_back = body_tform_chair_back.get_closest_se2_transform()
                        chair_center_rotations.append(se2_pose_to_np(body_tform_chair_back)[2])

                    if d.tag_id != CHAIR_CENTER_TAG_NUMBER:
                        chair_center = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                        cam_tform_chair_center = math_helpers.SE3Pose.from_matrix(chair_center)
                        body_tform_chair_center = body_tform_cam * cam_tform_chair_center
                        body_tform_chair_center = body_tform_chair_center.get_closest_se2_transform()
                        chair_center_rotations.append(se2_pose_to_np(body_tform_chair_center)[2])
                        chair_center_locations.append(se2_pose_to_np(body_tform_chair_center)[:2])

            final_im.append(im)

        if len(chair_center_locations) > 0 and len(chair_center_rotations) > 0:
            chair_center_rotations = np.mean(np.array(chair_center_rotations), axis=0).flatten()
            chair_center_locations = np.mean(np.array(chair_center_locations), axis=0).flatten()
            chair_state[0] = np.hstack((chair_center_locations, chair_center_rotations))

        if chair_state[0] is None and len(self.chair_states_in_body_frame) > 0:
            self.chair_states_in_body_frame.append(self.chair_states_in_body_frame[-1])
        else:
            if len(self.chair_states_in_body_frame) > 0 and self.chair_states_in_body_frame[-1][0] is not None:
                # self.chair_states_in_body_frame.append([0.75 * chair_state[0] + 0.25 * self.chair_states_in_body_frame[-1][0], chair_state[1]])
                self.chair_states_in_body_frame.append(chair_state)
                pass
            else:
                self.chair_states_in_body_frame.append(chair_state)

        img = np.concatenate(final_im, axis=-1)
        self.chair_images.append(img)

        if point_clouds_robot_perspective:
            point_clouds_robot_perspective = np.concatenate(point_clouds_robot_perspective, axis=0)
        self.point_clouds_robot_perspective.append(point_clouds_robot_perspective)

    def process_results(self):
        self.chair_images = []
        self.point_clouds_robot_perspective = []
        self.global_point_cloud_fid_frame = []
        self.chair_states_in_body_frame = []
        result_copy = self.results.copy()
        self.results = []
        for t, result in enumerate(result_copy):
            self.results.append(result)
            self.process_latest_result()

    def save_trajectory(self, image_file):
        self.process_results()
        traj = {'record_times': np.array(self.record_times),
                'chair_images': np.array(self.chair_images),
                'point_clouds': self.point_clouds_robot_perspective,
                'chair_states_in_body_frame': self.chair_states_in_body_frame}
        with open(image_file, 'wb') as f:
            pickle.dump(traj, f)



class OldAsyncRecordChairImageAndPose(AsyncPeriodicQuery):

    def __init__(self, image_client, image_sources, robot_state_task, online_mode = False):
        super(OldAsyncRecordChairImageAndPose, self).__init__('robot_state', image_client, LOGGER,
                                                           period_sec=TRAJECTORY_FRAME_RATE)
        self.step_counter = 0

        self.chair_images = []
        self.point_clouds = []
        self.record_times = []
        self.global_point_cloud_fid_frame = []

        self.image_sources = image_sources
        self.image_client = image_client

        self.robot_state_task = robot_state_task
        self.chair_states_in_body_frame = []

        options = apriltag.DetectorOptions(families="tag36h11")
        self.detector = apriltag.Detector(options)

        self.online_mode = online_mode

        self.results = []

    def _start_query(self):
        return self._client.get_image_from_sources_async(self.image_sources)

    def _handle_result(self, result):
        self.results.append(result)
        self.record_times.append(time.time() - REFERENCE_TIME)
        self.step_counter += 1
        if self.online_mode: self.process_latest_result()
        super(OldAsyncRecordChairImageAndPose, self)._handle_result(result)

    def process_latest_result(self):
        result = self.results[-1]
        final_im = []
        point_clouds = []

        for i, source_name in enumerate(self.image_sources):
            image_response = result[i]
            image_capture, image_source = image_response.shot, image_response.source
            focal_length, principal_point = image_source.pinhole.intrinsics.focal_length, image_source.pinhole.intrinsics.principal_point
            camera_params = [focal_length.x, focal_length.y, principal_point.x, principal_point.y]

            # print(image_capture.transforms_snapshot, image_capture.frame_name_image_sensor)
            if 'depth' in source_name:
                point_cloud = bosdyn.client.image.depth_image_to_pointcloud(image_response)
                camera_coordinates = get_a_tform_b(image_capture.transforms_snapshot, BODY_FRAME_NAME,
                                                   image_capture.frame_name_image_sensor)
                point_clouds.append(camera_coordinates.transform_cloud(point_cloud))

                im = np.frombuffer(image_capture.image.data, dtype=np.uint16)
                im = im.reshape(image_capture.image.rows,
                                image_capture.image.cols, 1)
                im = ((im - np.min(im)) / (np.max(im) - np.min(im)) * 255).astype(np.uint8)
            else:
                im = cv2.imdecode(np.frombuffer(image_capture.image.data, dtype=np.uint8), -1)
                im = im if len(im.shape) == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

            chair_state = [None] * 2

            if 'depth' not in source_name:
                detections, dimg = self.detector.detect(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), return_image=True)
                im = im // 2 + dimg[:, :, None] // 2
                body_tform_cam = get_a_tform_b(self.robot_state_task.results[-1].kinematic_state.transforms_snapshot,
                                               GRAV_ALIGNED_BODY_FRAME_NAME,
                                               BODY_FRAME_NAME) * get_a_tform_b(image_capture.transforms_snapshot,
                                                                                BODY_FRAME_NAME,
                                                                                image_capture.frame_name_image_sensor)
                for d in detections:
                    # print(d.tag_family, d.tag_id)
                    if d.tag_id != 10 or d.tag_family!=b'tag36h11':
                        continue
                    # print(d.center)
                    if d.center[0] < 240:
                        chair_back = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                        cam_tform_chair_back = math_helpers.SE3Pose.from_matrix(chair_back)
                        body_tform_chair_back = body_tform_cam * cam_tform_chair_back
                        body_tform_chair_back = body_tform_chair_back.get_closest_se2_transform()
                        if chair_state[0] is None:
                            chair_state[0] = se2_pose_to_np(body_tform_chair_back)
                        else:
                            chair_state[0] += se2_pose_to_np(body_tform_chair_back)
                            chair_state[0] /= 2

                    if d.center[0] > 240:
                        chair_leg = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                        cam_tform_chair_leg = math_helpers.SE3Pose.from_matrix(chair_leg)
                        body_tform_chair_leg = body_tform_cam * cam_tform_chair_leg
                        body_tform_chair_leg = body_tform_chair_leg.get_closest_se2_transform()
                        if 'frontright' in source_name: chair_state[1] = se2_pose_to_np(body_tform_chair_leg)

                if chair_state[0] is None and len(self.chair_states_in_body_frame) > 0:
                    self.chair_states_in_body_frame.append(self.chair_states_in_body_frame[-1])
                else:
                    if len(self.chair_states_in_body_frame) > 0 and self.chair_states_in_body_frame[-1][0] is not None:
                        self.chair_states_in_body_frame.append([0.5 * chair_state[0] + 0.5*self.chair_states_in_body_frame[-1][0], chair_state[1]])
                    else: self.chair_states_in_body_frame.append(chair_state)
                pass
            final_im.append(im)

        img = np.concatenate(final_im, axis=-1)
        self.chair_images.append(img)

        if point_clouds: point_clouds = np.concatenate(point_clouds, axis=0)
        self.point_clouds.append(point_clouds)

    def process_results(self):
        self.chair_images = []
        self.point_clouds = []
        self.global_point_cloud_fid_frame = []
        self.chair_states_in_body_frame = []
        for t, result in enumerate(self.results):
            final_im = []
            point_clouds = []

            for i, source_name in enumerate(self.image_sources):
                image_response = result[i]
                image_capture, image_source = image_response.shot, image_response.source
                focal_length, principal_point = image_source.pinhole.intrinsics.focal_length, image_source.pinhole.intrinsics.principal_point
                camera_params = [focal_length.x, focal_length.y, principal_point.x, principal_point.y]

                # print(image_capture.transforms_snapshot, image_capture.frame_name_image_sensor)
                if 'depth' in source_name:
                    point_cloud = bosdyn.client.image.depth_image_to_pointcloud(image_response)
                    camera_coordinates_robot = get_a_tform_b(image_capture.transforms_snapshot, BODY_FRAME_NAME,
                                                       image_capture.frame_name_image_sensor)
                    camera_coordinates_odom = get_a_tform_b(image_capture.transforms_snapshot, ODOM_FRAME_NAME,
                                                            image_capture.frame_name_image_sensor)
                    camera_coordinates_fid = self.robot_state_task.odom_tform_fid_3d.inverse() * camera_coordinates_odom
                    point_clouds.append(camera_coordinates_robot.transform_cloud(point_cloud))
                    new_pc = []
                    # for p in point_cloud:
                    #     if np.linalg.norm(np.asarray(p)) < 1:
                    #         new_pc.append(p)
                    # self.global_point_cloud_fid_frame.extend(camera_coordinates_fid.transform_cloud(new_pc))
                    self.global_point_cloud_fid_frame.extend(camera_coordinates_fid.transform_cloud(point_cloud))

                    im = np.frombuffer(image_capture.image.data, dtype=np.uint16)
                    im = im.reshape(image_capture.image.rows,
                                    image_capture.image.cols, 1)
                    im = ((im - np.min(im)) / (np.max(im) - np.min(im)) * 255).astype(np.uint8)
                else:
                    im = cv2.imdecode(np.frombuffer(image_capture.image.data, dtype=np.uint8), -1)
                    im = im if len(im.shape) == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

                chair_state = [None] * 2
                chair_state[1] = np.zeros((3,))

                if 'depth' not in source_name:
                    detections, dimg = self.detector.detect(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), return_image=True)
                    im = im // 2 + dimg[:, :, None] // 2
                    body_tform_cam = get_a_tform_b(self.robot_state_task.results[t].kinematic_state.transforms_snapshot,
                                                GRAV_ALIGNED_BODY_FRAME_NAME,
                                                BODY_FRAME_NAME) * get_a_tform_b(image_capture.transforms_snapshot,
                                                                                    BODY_FRAME_NAME,
                                                                                    image_capture.frame_name_image_sensor)
                    for d in detections:
                        if d.center[0] < 240:
                            chair_back = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                            cam_tform_chair_back = math_helpers.SE3Pose.from_matrix(chair_back)
                            body_tform_chair_back = body_tform_cam * cam_tform_chair_back
                            body_tform_chair_back = body_tform_chair_back.get_closest_se2_transform()
                            if chair_state[0] is None:
                                chair_state[0] = se2_pose_to_np(body_tform_chair_back)
                            else:
                                chair_state[0] += se2_pose_to_np(body_tform_chair_back)
                                chair_state[0] /= 2

                        if d.center[0] > 240:
                            chair_leg = self.detector.detection_pose(d, camera_params, tag_size=APRIL_TAG_SIZE)[0]
                            cam_tform_chair_leg = math_helpers.SE3Pose.from_matrix(chair_leg)
                            body_tform_chair_leg = body_tform_cam * cam_tform_chair_leg
                            body_tform_chair_leg = body_tform_chair_leg.get_closest_se2_transform()
                            if 'frontright' in source_name: 
                                chair_state[1] = se2_pose_to_np(body_tform_chair_leg)

                    if chair_state[0] is None and len(self.chair_states_in_body_frame) > 0:
                        self.chair_states_in_body_frame.append(self.chair_states_in_body_frame[-1])
                    else:
                        if len(self.chair_states_in_body_frame) > 0 and self.chair_states_in_body_frame[-1][0] is not None:
                            self.chair_states_in_body_frame.append([0.5 * chair_state[0] + 0.5*self.chair_states_in_body_frame[-1][0], chair_state[1]])
                        else: self.chair_states_in_body_frame.append(chair_state)
                final_im.append(im)

            img = np.concatenate(final_im, axis=-1)
            self.chair_images.append(img)

            if point_clouds: point_clouds = np.concatenate(point_clouds, axis=0)
            self.point_clouds.append(point_clouds)

    def save_trajectory(self, image_file):
        self.process_results()
        traj = {'record_times': np.array(self.record_times),
                'chair_images': np.array(self.chair_images),
                'point_clouds': self.point_clouds,
                'chair_states_in_body_frame': self.chair_states_in_body_frame}
        with open(image_file, 'wb') as f:
            pickle.dump(traj, f)

