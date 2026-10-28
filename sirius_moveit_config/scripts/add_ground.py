#!/usr/bin/env python3
"""Add the ground to the MoveIt planning scene (it knows nothing about the Gazebo world).

Without it OMPL may route the arm through the ground; in Gazebo the arm then hits it
and the trajectory is aborted. The plane is attached to base_link (flat terrain), its
top sits `margin` above the real ground, and contact between the ground and the rover
body (wheels on the ground) is allowed so only the arm is kept away from it. The same
holds for the octomap (moveit.launch.py octomap:=true): the wheels stand on mapped ground.
"""

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (AllowedCollisionEntry, CollisionObject, PlanningScene,
                             PlanningSceneComponents)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from shape_msgs.msg import SolidPrimitive

GROUND = 'ground'
OCTOMAP = '<octomap>'           # MoveIt's name of the octomap in the collision matrix


def _call(node, client, request):
    while not client.wait_for_service(timeout_sec=2.0):
        node.get_logger().info(f'Waiting for {client.srv_name} (move_group)...')
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    return future.result()


def _allow_ground(acm, body_links, objects=(GROUND, OCTOMAP)):
    """Return a copy of the ACM with rows/columns for objects (allowed only for body links)."""
    names = list(acm.entry_names)
    for obj in objects:
        if obj in names:
            continue
        for name, entry in zip(names, acm.entry_values):
            entry.enabled.append(name in body_links)
        names.append(obj)
        acm.entry_values.append(AllowedCollisionEntry(
            enabled=[name in body_links for name in names[:-1]] + [False]))
    acm.entry_names = names
    return acm


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('add_ground')
    frame = node.declare_parameter('frame', 'base_link').value
    size = node.declare_parameter('size', 20.0).value
    margin = node.declare_parameter('margin', 0.01).value  # [m] kept between the arm and ground
    body_links = set(node.declare_parameter('body_links', [
        'platform', 'front_camera', 'mast_front_camera', 'mast_back_camera', 'gnss',
        'indicator_light',
        *[f'{side}_{part}' for side in ('front_left', 'front_right', 'back_left', 'back_right')
          for part in ('suspension', 'wheel', 'inner_rim', 'outer_rim')],
    ]).value)
    thickness = 0.1

    # A non-empty ACM in a scene diff replaces the whole matrix: start from the current one.
    current = _call(node, node.create_client(GetPlanningScene, '/get_planning_scene'),
                    GetPlanningScene.Request(components=PlanningSceneComponents(
                        components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX)))

    ground = CollisionObject(id=GROUND, operation=CollisionObject.ADD)
    ground.header.frame_id = frame
    ground.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[size, size, thickness])]
    pose = Pose()
    pose.position.z = margin - thickness / 2
    pose.orientation.w = 1.0
    ground.primitive_poses = [pose]

    scene = PlanningScene(is_diff=True)
    scene.world.collision_objects = [ground]
    scene.allowed_collision_matrix = _allow_ground(
        current.scene.allowed_collision_matrix, body_links)
    result = _call(node, node.create_client(ApplyPlanningScene, '/apply_planning_scene'),
                   ApplyPlanningScene.Request(scene=scene))
    if result is not None and result.success:
        node.get_logger().info('Ground added to the planning scene')
    else:
        node.get_logger().error('Failed to add the ground to the planning scene')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
