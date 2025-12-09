from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import os
from pxr import Usd, UsdPhysics, Sdf, PhysxSchema

usd_path = os.path.abspath("/workspace/isaaclab/assets/Factory_new/industreal_round_hole_16mm.usd")
print(f"Fixing hole USD file: {usd_path}")

stage = Usd.Stage.Open(usd_path)
if stage is None:
    print("Error: Failed to open USD stage")
    simulation_app.close()
    exit(1)

root_layer = stage.GetRootLayer()
pseudo_root = stage.GetPseudoRoot()

root_prim = stage.GetPrimAtPath("/Root")
if not root_prim or not root_prim.IsValid():
    root_prim = stage.DefinePrim("/Root", "Xform")
    print("Created /Root prim")

stage.SetDefaultPrim(root_prim)
print(f"Default prim set to: {root_prim.GetPath()}")

for child in list(root_prim.GetChildren()):
    stage.RemovePrim(child.GetPath())
    print(f"Removed existing child under /Root: {child.GetPath()}")

for prim in list(pseudo_root.GetChildren()):
    name = prim.GetName()
    path = prim.GetPath()
    if name in ["Root", "Looks", "Materials", "World"]:
        continue
    new_path = Sdf.Path(f"/Root/{name}")
    if not root_layer.GetPrimAtPath(new_path):
        Sdf.CopySpec(root_layer, path, root_layer, new_path)
        print(f"Copied top-level prim {path} to {new_path}")
    stage.RemovePrim(path)
    print(f"Removed old top-level prim: {path}")

root_prim = stage.GetPrimAtPath("/Root")
if root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
    root_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    print("Removed ArticulationRootAPI from /Root")

tray_path = "/Root/industreal_tray_insert_round_peg_16mm"
tray_prim = stage.GetPrimAtPath(tray_path)
if not tray_prim or not tray_prim.IsValid():
    print(f"Error: {tray_path} prim not found")
    simulation_app.close()
    exit(1)

if not UsdPhysics.RigidBodyAPI.Get(stage, tray_path):
    UsdPhysics.RigidBodyAPI.Apply(tray_prim)
    print(f"Applied RigidBodyAPI to {tray_path}")
else:
    print(f"RigidBodyAPI already present on {tray_path}")

if not UsdPhysics.ArticulationRootAPI.Get(stage, tray_path):
    UsdPhysics.ArticulationRootAPI.Apply(tray_prim)
    print(f"Applied ArticulationRootAPI to {tray_path}")
else:
    print(f"ArticulationRootAPI already present on {tray_path}")

collision_prims = []
for prim in stage.Traverse():
    if UsdPhysics.CollisionAPI.Get(stage, prim.GetPath()):
        collision_prims.append(prim)

if not collision_prims:
    print("No collision prims found to apply SDF.")
else:
    for prim in collision_prims:
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        approx_attr = mesh_collision_api.GetApproximationAttr()
        if not approx_attr:
            approx_attr = mesh_collision_api.CreateApproximationAttr()
        approx_attr.Set("sdf")
        print(f"Set SDF approximation on {prim.GetPath()}")

stage.Save()

from pxr import UsdPhysics as _UsdPhysics

def print_tree(prim, indent=0):
    t = prim.GetTypeName()
    path = prim.GetPath()
    flags = []
    if _UsdPhysics.RigidBodyAPI.Get(stage, path):
        flags.append("RigidBody")
    if _UsdPhysics.ArticulationRootAPI.Get(stage, path):
        flags.append("ArtRoot")
    if _UsdPhysics.CollisionAPI.Get(stage, path):
        flags.append("Collision")
    if _UsdPhysics.MassAPI.Get(stage, path):
        flags.append("Mass")
    if prim.IsInstanceable():
        flags.append("Instanceable")
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, path)
    if mesh_collision_api:
        approx = mesh_collision_api.GetApproximationAttr()
        if approx and approx.HasAuthoredValueOpinion():
            flags.append(f"Approx={approx.Get()}")
    flag_str = f" [{' ,'.join(flags)}]" if flags else ""
    print("  " * indent + f"{path} ({t}){flag_str}")
    for c in prim.GetChildren():
        print_tree(c, indent + 1)

print("\n" + "="*80)
print(f"✓ Successfully updated {os.path.basename(usd_path)}")
print("  - /Root is defaultPrim")
print("  - All asset prims moved under /Root/<asset>")
print("  - /Root/industreal_tray_insert_round_peg_16mm is RigidBody + ArtRoot")
print("  - Collision prims set to SDF approximation where present")
print("="*80)

print("\nUSD PRIM TREE AFTER FIX:\n")
print_tree(stage.GetPseudoRoot())

simulation_app.close()
