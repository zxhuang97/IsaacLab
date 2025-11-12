from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.rotations as rotations_utils
import numpy as np
from pxr import UsdPhysics, Sdf, Gf, Usd, UsdGeom
import isaacsim.core.utils.prims as prims_utils
import os
import sys

# Configuration
FINGER_TYPE = "mini"  # Options: "r15" or "mini"

franka_url = os.path.abspath("assets/Factory_new/franka_mimic_ori.usd")

if FINGER_TYPE == "r15":
    finger_use_url = os.path.abspath("assets/TacSL/gelsight_r15_finger.usd")
    output_usd_path = "assets/Factory_new/franka_gelsight_r15_assembled.usd"
    fingertip_z_offset = 0.135
elif FINGER_TYPE == "mini":
    finger_use_url = os.path.abspath("assets/TacSL/gsmini_finger.usd")
    fingertip_z_offset = 0.13
    finger_x_offset = 0.01
    output_usd_path = f"assets/Factory_new/franka_gelsight_mini_assembled_z{int(fingertip_z_offset*100)}_x{int(finger_x_offset*1000)}.usd"
else:
    raise ValueError(f"Unknown finger type: {FINGER_TYPE}")

print(f"Using finger type: {FINGER_TYPE}")
print(f"Franka URL: {franka_url}")
print(f"Finger URL: {finger_use_url}")
print(f"Output URL: {output_usd_path}")

stage_utils.create_new_stage()
stage_utils.open_stage(franka_url)
stage = stage_utils.get_current_stage()

# Remove existing fingers
stage.RemovePrim("/panda/panda_leftfinger")
stage.RemovePrim("/panda/panda_rightfinger")

left_prim_path = "/panda/panda_leftfinger"
right_prim_path = "/panda/panda_rightfinger"
hand_xform = "/panda/panda_hand"

left_joint_path = "/panda/panda_hand/panda_finger_joint1"
right_joint_path = "/panda/panda_hand/panda_finger_joint2"

right_joint_prim = prims_utils.get_prim_at_path(right_joint_path)

# Modify panda_fingertip_centered_joint z-position
fingertip_joint_path = "/panda/panda_hand/panda_fingertip_centered_joint"
fingertip_joint_prim = prims_utils.get_prim_at_path(fingertip_joint_path)
if fingertip_joint_prim and fingertip_joint_prim.IsValid():
    fingertip_joint = UsdPhysics.PrismaticJoint(fingertip_joint_prim)
    local_pos_0 = fingertip_joint.GetLocalPos0Attr().Get()
    print(f"local_pos_0: {local_pos_0}")
    if local_pos_0 is not None:
        new_local_pos_0 = Gf.Vec3f(local_pos_0[0], local_pos_0[1], fingertip_z_offset)
        fingertip_joint.GetLocalPos0Attr().Set(new_local_pos_0)
        print(f"Updated panda_fingertip_centered_joint localPos0 z from {local_pos_0[2]} to {fingertip_z_offset}")

# Add finger references
stage_utils.add_reference_to_stage(finger_use_url, left_prim_path)
stage_utils.add_reference_to_stage(finger_use_url, right_prim_path)

hand_prim = prims_utils.get_prim_at_path(hand_xform)
leftfinger_prim_path = left_prim_path + '/gelsight_finger'
rightfinger_prim_path = right_prim_path + '/gelsight_finger'

left_finger_prim = prims_utils.get_prim_at_path(leftfinger_prim_path)
right_finger_prim = prims_utils.get_prim_at_path(rightfinger_prim_path)

# Configure left joint
left_joint_prim = prims_utils.get_prim_at_path(left_joint_path)
left_joint = UsdPhysics.PrismaticJoint(left_joint_prim)
left_joint.GetBody0Rel().SetTargets([hand_prim.GetPath()])
left_joint.GetBody1Rel().SetTargets([left_finger_prim.GetPath()])


# Configure right joint
right_joint = UsdPhysics.PrismaticJoint(right_joint_prim)
right_joint.GetBody0Rel().SetTargets([hand_prim.GetPath()])
right_joint.GetBody1Rel().SetTargets([right_finger_prim.GetPath()])

# Set local translation for right joint from URDF (origin xyz="0 0.02 0.06340285")
if FINGER_TYPE == "mini":
    left_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, finger_x_offset, 0.06340285))
    print(f"Set left joint localPos0 to (0.0, {finger_x_offset}, 0.06340285)")
    right_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, -finger_x_offset, 0.06340285))
    print(f"Set right joint localPos0 to (0.0, -{finger_x_offset}, 0.06340285)")

# Rotate right finger by 180 degrees to make it symmetrical
new_orient = rotations_utils.euler_angles_to_quat(np.array([180, 0.0, 180]), degrees=True, extrinsic=False)
right_joint.GetLocalRot1Attr().Set(Gf.Quatf(*new_orient.astype(float)))

# Hide cameras if present
left_cam_path = "/panda/panda_leftfinger/gelsight_finger/elastomer_tip/cam"
right_cam_path = "/panda/panda_rightfinger/gelsight_finger/elastomer_tip/cam"

left_cam_prim = prims_utils.get_prim_at_path(left_cam_path)
right_cam_prim = prims_utils.get_prim_at_path(right_cam_path)

if left_cam_prim and left_cam_prim.IsValid():
    left_cam_prim.GetAttribute("visibility").Set("invisible")
if right_cam_prim and right_cam_prim.IsValid():
    right_cam_prim.GetAttribute("visibility").Set("invisible")

# Export the assembled stage as a flattened USD
temp_layer = Sdf.Layer.CreateNew(output_usd_path)
temp_stage = Usd.Stage.Open(temp_layer)

# Update stage metadata
UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.GetStageUpAxis(stage))
UsdGeom.SetStageMetersPerUnit(temp_stage, UsdGeom.GetStageMetersPerUnit(stage))

# Copy the root prim to the new stage
source_layer = stage.GetRootLayer()
Sdf.CreatePrimInLayer(temp_layer, "/")
Sdf.CopySpec(source_layer, "/", temp_layer, "/")

# Set the default prim
temp_layer.defaultPrim = "panda"

# Save and flatten the stage
temp_stage.Flatten()

# Now rename the gelsight_finger prims in the flattened stage
# After flattening, the prims are no longer from references, so we can rename them
left_gelsight_spec = temp_layer.GetPrimAtPath("/panda/panda_leftfinger/gelsight_finger")
right_gelsight_spec = temp_layer.GetPrimAtPath("/panda/panda_rightfinger/gelsight_finger")

if left_gelsight_spec:
    left_gelsight_spec.name = "panda_leftfinger"
    print(f"Renamed left gelsight_finger to panda_leftfinger")

if right_gelsight_spec:
    right_gelsight_spec.name = "panda_rightfinger"
    print(f"Renamed right gelsight_finger to panda_rightfinger")

temp_layer.Save()

print(f"Flattened USD exported to: {output_usd_path}")