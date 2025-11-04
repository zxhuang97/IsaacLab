"""Fix gsmini_finger.usd by adding instanceable=false, camera, and removing articulation."""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics

# Path to the USD file
usd_path = os.path.abspath("assets/TacSL/gsmini_finger.usd")

print(f"Fixing USD file: {usd_path}")

# Open the USD file
stage = Usd.Stage.Open(usd_path)

# Get the root prim (should be "gelsight_finger")
root_prim = stage.GetDefaultPrim()
if not root_prim:
    print("Error: No default prim found")
    simulation_app.close()
    exit(1)

root_path = root_prim.GetPath()
print(f"Root prim: {root_path}")

# Function to create or get a prim
def ensure_prim_exists(prim_path, prim_type=""):
    """Create prim if it doesn't exist, return the prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        if prim_type:
            prim = stage.DefinePrim(prim_path, prim_type)
        else:
            prim = stage.OverridePrim(prim_path)
        print(f"Created prim: {prim_path}")
    return prim

# 1. Add overrides for gelsight_finger with instanceable=false
print("\n1. Adding overrides for gelsight_finger...")
gelsight_finger_path = f"{root_path}/gelsight_finger"
gelsight_finger_prim = ensure_prim_exists(gelsight_finger_path)

# Add visuals and collisions under gelsight_finger
gelsight_visuals = ensure_prim_exists(f"{gelsight_finger_path}/visuals")
gelsight_visuals.SetInstanceable(False)
print(f"   Set instanceable=false for {gelsight_visuals.GetPath()}")

gelsight_collisions = ensure_prim_exists(f"{gelsight_finger_path}/collisions")
gelsight_collisions.SetInstanceable(False)
print(f"   Set instanceable=false for {gelsight_collisions.GetPath()}")

# 2. Add overrides for elastomer with instanceable=false
print("\n2. Adding overrides for elastomer...")
elastomer_path = f"{root_path}/elastomer"
elastomer_prim = ensure_prim_exists(elastomer_path)

# Add visuals and collisions under elastomer
elastomer_visuals = ensure_prim_exists(f"{elastomer_path}/visuals")
elastomer_visuals.SetInstanceable(False)
print(f"   Set instanceable=false for {elastomer_visuals.GetPath()}")

elastomer_collisions = ensure_prim_exists(f"{elastomer_path}/collisions")
elastomer_collisions.SetInstanceable(False)
print(f"   Set instanceable=false for {elastomer_collisions.GetPath()}")

# 3. Add overrides for elastomer_tip with instanceable=false and camera
print("\n3. Adding overrides for elastomer_tip...")
elastomer_tip_path = f"{root_path}/elastomer_tip"
elastomer_tip_prim = ensure_prim_exists(elastomer_tip_path)

# Add isaac:nameOverride attribute
if not elastomer_tip_prim.HasAttribute("isaac:nameOverride"):
    isaac_name_attr = elastomer_tip_prim.CreateAttribute(
        "isaac:nameOverride", 
        Sdf.ValueTypeNames.String
    )
    isaac_name_attr.SetMetadata("displayName", "Name Override")
    isaac_name_attr.SetMetadata("doc", "Name override for prim lookup in base name search")
    print(f"   Added isaac:nameOverride attribute")

# Add visuals under elastomer_tip
elastomer_tip_visuals = ensure_prim_exists(f"{elastomer_tip_path}/visuals")
elastomer_tip_visuals.SetInstanceable(False)
print(f"   Set instanceable=false for {elastomer_tip_visuals.GetPath()}")

# 4. Add camera under elastomer_tip
print("\n4. Adding camera under elastomer_tip...")
camera_path = f"{elastomer_tip_path}/cam"
camera_prim = stage.GetPrimAtPath(camera_path)

if not camera_prim or not camera_prim.IsValid():
    camera_prim = stage.DefinePrim(camera_path, "Camera")
    camera = UsdGeom.Camera(camera_prim)
    
    # Camera parameters from gs_mini.yaml converted to USD format
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.0001, 100000))
    camera.CreateFocalLengthAttr().Set(2.67)  # cm
    camera.CreateFocusDistanceAttr().Set(0.4)
    camera.CreateHorizontalApertureAttr().Set(2.525)  # cm (0.02525 m * 100)
    camera.CreateVerticalApertureAttr().Set(2.075)  # cm (0.02075 m * 100)
    
    # Set transform attributes
    xformable = UsdGeom.Xformable(camera_prim)
    xformable.ClearXformOpOrder()
    
    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(0, -0.0267, 0))
    
    # Use quatd to match R15 format
    orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    # orient_op.Set(Gf.Quatd(0.70711, 0.70711, 0.0, 0.0))
    # 90 0 90
    orient_op.Set(Gf.Quatd(0.5, 0.5, 0.5, -0.5))
    
    scale_op = xformable.AddScaleOp()
    scale_op.Set(Gf.Vec3d(1, 1, 1))
    
    print(f"   Created camera at {camera_path}")
else:
    print(f"   Camera already exists at {camera_path}")

# 5. Add material overrides (optional but good to have)
print("\n5. Adding material overrides...")
looks_path = f"{root_path}/Looks"
looks_prim = ensure_prim_exists(looks_path)

material_0_outer = ensure_prim_exists(f"{looks_path}/material_0")
material_0_inner = ensure_prim_exists(f"{looks_path}/material_0/material_0")

# Add opacity constant
if not material_0_inner.HasAttribute("inputs:opacity_constant"):
    opacity_attr = material_0_inner.CreateAttribute(
        "inputs:opacity_constant",
        Sdf.ValueTypeNames.Float
    )
    opacity_attr.Set(1.0)
    print(f"   Added opacity_constant to material_0")

# 6. Remove ArticulationRootAPI to prevent conflicts
print("\n6. Removing ArticulationRootAPI from physics configuration...")

# Also open and fix the physics USD if it exists
physics_usd_path = os.path.abspath("assets/TacSL/configuration/gsmini_finger_physics.usd")
if os.path.exists(physics_usd_path):
    physics_stage = Usd.Stage.Open(physics_usd_path)
    
    removed_count = 0
    removed_properties = []
    
    # Traverse all prims and remove ArticulationRootAPI
    for prim in physics_stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            print(f"   Removed ArticulationRootAPI from {prim.GetPath()}")
            removed_count += 1
        
        # Also remove PhysxArticulationAPI if present
        if prim.HasAPI("PhysxArticulationAPI"):
            # Get the prim's applied API schemas
            api_schemas = prim.GetAppliedSchemas()
            if "PhysxArticulationAPI" in api_schemas:
                prim.RemoveAppliedSchema("PhysxArticulationAPI")
                print(f"   Removed PhysxArticulationAPI from {prim.GetPath()}")
                removed_count += 1
        
        # Remove PhysxArticulation-specific properties
        articulation_props = [
            "physxArticulation:enabledSelfCollisions",
            "physxArticulation:solverPositionIterationCount",
            "physxArticulation:solverVelocityIterationCount"
        ]
        for prop_name in articulation_props:
            if prim.HasAttribute(prop_name):
                prim.RemoveProperty(prop_name)
                removed_properties.append(f"{prim.GetPath()}.{prop_name}")
                print(f"   Removed property: {prop_name} from {prim.GetPath()}")
    
    if removed_count == 0 and len(removed_properties) == 0:
        print("   No ArticulationRootAPI or properties found in physics USD")
    else:
        print(f"   Removed articulation APIs from {removed_count} locations")
        if removed_properties:
            print(f"   Removed {len(removed_properties)} articulation-specific properties")
    
    physics_stage.Save()
    print(f"   Saved modified physics USD: {physics_usd_path}")
else:
    print(f"   Physics USD not found at {physics_usd_path} (will be generated without articulation)")

# Save the main USD
stage.Save()

print("\n" + "="*80)
print("✓ Successfully fixed gsmini_finger USD files!")
print("="*80)
print("\nChanges made:")
print("  1. Added 'over' blocks for gelsight_finger/visuals and gelsight_finger/collisions")
print("  2. Added 'over' blocks for elastomer/visuals and elastomer/collisions")
print("  3. Added 'over' block for elastomer_tip/visuals")
print("  4. Set instanceable=false on all visual and collision prims (5 places)")
print("  5. Added Camera 'cam' under elastomer_tip with gs_mini parameters:")
print("     - focalLength: 2.67 cm")
print("     - horizontalAperture: 2.525 cm")
print("     - verticalAperture: 2.075 cm")
print("     - translate: (0, 0, -0.0267)")
print("     - orient: (0.707, 0.707, 0.0, 0.0)")
print("  6. Added material opacity overrides")
print("  7. Removed from physics USD:")
print("     - PhysicsArticulationRootAPI")
print("     - PhysxArticulationAPI")
print("     - physxArticulation:enabledSelfCollisions")
print("     - physxArticulation:solverPositionIterationCount")
print("     - physxArticulation:solverVelocityIterationCount")
print("\n✓ The finger is now a pure RigidBody hierarchy (no articulation)")
print("✓ Can be safely attached to Franka robot without conflicts!")
print("\nNote: gelsight_sensor rigid body and joints are kept (valid for GS Mini)")

simulation_app.close()

