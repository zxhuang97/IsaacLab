from isaaclab.app import AppLauncher
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.rotations as rotations_utils
import numpy as np
from pxr import UsdPhysics, Sdf, Gf, PhysxSchema, Usd, UsdGeom, UsdUtils
import isaacsim.core.utils.prims as prims_utils
import omni
import os

stage_utils.create_new_stage()
# franka_url = "/home/zixuanh/force_tool_tactile/assets/Factory_new/franka_mimic_ori.usd"
# finger_use_url = "/home/zixuanh/force_tool_tactile/assets/TacSL//gelsight_r15_finger.usd"

franka_url = os.path.abspath("assets/Factory_new/franka_mimic_ori.usd")
finger_use_url = os.path.abspath("assets/TacSL/gelsight_r15_finger.usd")


stage_utils.open_stage(franka_url)
stage = stage_utils.get_current_stage()
stage.RemovePrim("/panda/panda_leftfinger")
stage.RemovePrim("/panda/panda_rightfinger")


left_prim_path = "/panda/panda_leftfinger"
right_prim_path = "/panda/panda_rightfinger"
hand_xform = "/panda/panda_hand"
robot_prim_path = "/panda"

left_joint_path = "/panda/panda_hand/panda_finger_joint1"
right_joint_path = "/panda/panda_hand/panda_finger_joint2"

right_joint_prim =  prims_utils.get_prim_at_path(right_joint_path)

# Modify panda_fingertip_centered_joint z-position
fingertip_joint_path = "/panda/panda_hand/panda_fingertip_centered_joint"
fingertip_joint_prim = prims_utils.get_prim_at_path(fingertip_joint_path)
if fingertip_joint_prim and fingertip_joint_prim.IsValid():
    fingertip_joint = UsdPhysics.PrismaticJoint(fingertip_joint_prim)
    # Get the current localPos0 and modify the z value
    local_pos_0 = fingertip_joint.GetLocalPos0Attr().Get()
    print(f"local_pos_0: {local_pos_0}")
    if local_pos_0 is not None:
        new_local_pos_0 = Gf.Vec3f(local_pos_0[0], local_pos_0[1], 0.135)
        fingertip_joint.GetLocalPos0Attr().Set(new_local_pos_0)
        print(f"Updated panda_fingertip_centered_joint localPos0 z from {local_pos_0[2]} to 0.135")

stage_utils.add_reference_to_stage(finger_use_url, left_prim_path)
stage_utils.add_reference_to_stage(finger_use_url, right_prim_path)
hand_prim = prims_utils.get_prim_at_path(hand_xform)
leftfinger_prim_path = left_prim_path + '/gelsight_finger'
rightfinger_prim_path = right_prim_path + '/gelsight_finger'

left_finger_prim = prims_utils.get_prim_at_path(leftfinger_prim_path)
right_finger_prim = prims_utils.get_prim_at_path(rightfinger_prim_path)

left_joint_prim =  prims_utils.get_prim_at_path(left_joint_path)
left_joint = UsdPhysics.PrismaticJoint(left_joint_prim)

left_joint.GetBody0Rel().SetTargets([hand_prim.GetPath()])
left_joint.GetBody1Rel().SetTargets([left_finger_prim.GetPath()])

right_joint = UsdPhysics.PrismaticJoint(right_joint_prim)
right_joint.GetBody0Rel().SetTargets([hand_prim.GetPath()])
right_joint.GetBody1Rel().SetTargets([right_finger_prim.GetPath()])

# because we have one finger, so rotate the right joint by 180 rot-z to make it symmetrical to the left finger
new_orient = rotations_utils.euler_angles_to_quat(np.array([180 ,0.0, 180]), degrees=True, extrinsic=False)
right_joint.GetLocalRot1Attr().Set(Gf.Quatf(*new_orient.astype(float)))

prims_utils.get_prim_at_path("/panda/panda_rightfinger/elastomer_tip/cam").GetAttribute("visibility").Set("invisible")
prims_utils.get_prim_at_path("/panda/panda_leftfinger/elastomer_tip/cam").GetAttribute("visibility").Set("invisible")

# Export the assembled stage as a flattened USD
output_usd_path = "assets/Factory_new/franka_gelsight_r15_assembled.usd"

# Create a new stage for the flattened output
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
temp_layer.Save()

print(f"Flattened USD exported to: {output_usd_path}")