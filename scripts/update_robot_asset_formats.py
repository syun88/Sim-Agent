"""Synchronize repaired convex geometry into USD and Gazebo SDF.

Run with: uv run --with usd-core python scripts/update_robot_asset_formats.py FOLDER
The Gazebo adapter attaches collision-only fixed children to avoid direct SDF
adjacency between the collision bodies. Fixed-link handling must still be
validated in the actual Gazebo engine; this script checks structure only.
"""
import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import trimesh


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    folder=parser.parse_args().folder.resolve()
    geometry=json.loads((folder/'source/collision_repair_geometry.json').read_text())
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Vt
    path=folder/'isaac_sim/scorpion.usda';stage=Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            prim.RemoveProperty('physics:filteredPairs');prim.RemoveAPI(UsdPhysics.FilteredPairsAPI)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateCollisionEnabledAttr(True)
    count=0
    for link,names in geometry.items():
        parent=stage.GetPrimAtPath('/Scorpion/'+link)
        for child in list(parent.GetChildren()):
            if child.GetName().startswith('collision_'):stage.RemovePrim(child.GetPath())
        for i,name in enumerate(names):
            source=trimesh.load(folder/'genesis'/name,force='mesh',process=False)
            mesh=UsdGeom.Mesh.Define(stage,parent.GetPath().AppendChild('collision_'+str(i)))
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(source.vertices,np.float32)))
            mesh.CreateFaceVertexCountsAttr([3]*len(source.faces))
            mesh.CreateFaceVertexIndicesAttr(np.asarray(source.faces,np.int32).reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateVisibilityAttr('invisible');mesh.CreatePurposeAttr('guide')
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
            mesh.GetPrim().CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.0002)
            mesh.GetPrim().CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
            count+=1
    stage.GetRootLayer().Save();stage.GetRootLayer().Export(str(folder/'isaac_sim/scorpion.usd'))
    # Always build the adapter from the original SDF; do not accumulate proxies.
    backup=folder.parent/'old_no_use/for_sim_collision_before_20261021.zip'
    with zipfile.ZipFile(backup) as archive:
        root=ET.fromstring(archive.read('gazebo_jetty/models/scorpion/model.sdf'))
    model=root.find('model');template=copy.deepcopy(model.find('link/collision'))
    if model.find('self_collide') is None:ET.SubElement(model,'self_collide')
    model.find('self_collide').text='true'
    for link,names in geometry.items():
        real=model.find(f"link[@name='{link}']")
        for collision in list(real.findall('collision')):real.remove(collision)
        proxy_name='collision_proxy__'+link
        proxy=ET.SubElement(model,'link',name=proxy_name)
        ET.SubElement(proxy,'pose',relative_to=link).text='0 0 0 0 0 0'
        ET.SubElement(proxy,'self_collide').text='true'
        inertial=ET.SubElement(proxy,'inertial');ET.SubElement(inertial,'mass').text='1e-9'
        tensor=ET.SubElement(inertial,'inertia')
        for name in ('ixx','iyy','izz','ixy','ixz','iyz'):
            ET.SubElement(tensor,name).text='1e-12' if name in ('ixx','iyy','izz') else '0'
        for i,name in enumerate(names):
            collision=copy.deepcopy(template);collision.set('name',f'{link}_collision_{i}')
            pose=collision.find('pose')
            if pose is not None:pose.text='0 0 0 0 0 0'
            collision.find('geometry/mesh/uri').text='model://scorpion/'+name
            proxy.append(collision)
        joint=ET.SubElement(model,'joint',name='collision_fixed__'+link,type='fixed')
        ET.SubElement(joint,'parent').text=link;ET.SubElement(joint,'child').text=proxy_name
        ET.SubElement(joint,'pose',relative_to=link).text='0 0 0 0 0 0'
    ET.indent(root);ET.ElementTree(root).write(folder/'gazebo_jetty/models/scorpion/model.sdf',encoding='utf-8',xml_declaration=True)
    report={'usd_collision_meshes':count,'usd_joint_collisions_enabled':True,'usd_filtered_pairs':[],
            'gazebo_adapter':'collision-only fixed child per physical link; actual engine fixed-link handling unverified',
            'gazebo_collision_proxy_links':len(geometry),'gazebo_added_mass_kg':len(geometry)*1e-9,
            'gazebo_runtime_tested':False,'isaac_physx_runtime_tested':False}
    (folder/'source/collision_format_repair.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
