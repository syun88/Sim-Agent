"""Cross-format structure and provenance checks; no simulator motion is claimed."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def canonical(node):
    return (node.tag, sorted(node.attrib.items()), (node.text or '').strip(), [canonical(c) for c in node])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    folder=parser.parse_args().folder.resolve()
    geometry=json.loads((folder/'source/collision_repair_geometry.json').read_text())
    expected=sum(map(len,geometry.values()));checks={}
    with zipfile.ZipFile(folder.parent/'old_no_use/for_sim_collision_before_20261021.zip') as archive:
        old=ET.fromstring(archive.read('genesis/scorpion.urdf'))
        now=ET.parse(folder/'genesis/scorpion.urdf').getroot()
        checks['joint_frames_and_source_limits_unchanged']=[canonical(j) for j in old.findall('joint')]==[canonical(j) for j in now.findall('joint')]
        checks['inertias_unchanged']=[canonical(l.find('inertial')) for l in old.findall('link')]==[canonical(l.find('inertial')) for l in now.findall('link')]
        visual_paths=[p for p in folder.rglob('*') if p.is_file() and '/meshes/visual/' in str(p)]
        checks['visual_mesh_bytes_unchanged']=all(p.read_bytes()==archive.read(str(p.relative_to(folder))) for p in visual_paths)
    checks['urdf_collision_count']=len(now.findall('.//collision'))==expected
    checks['urdf_mesh_references_exist']=all((folder/'genesis'/mesh.get('filename')).is_file() for mesh in now.findall('.//mesh'))
    inventory=json.loads((folder/'source/model.json').read_text())
    checks['source_inventory_matches']={link['name']:link['collision'] for link in inventory['links']}==geometry
    directories=['urdf','mujoco','gazebo_jetty/models/scorpion']
    checks['collision_copies_identical']=all((folder/directory/name).read_bytes()==(folder/'genesis'/name).read_bytes()
        for names in geometry.values() for name in names for directory in directories)
    checks['srdf_has_no_pair_exclusions']=all(not ET.parse(folder/f).getroot().findall('disable_collisions') for f in ['genesis/scorpion.srdf','urdf/scorpion.srdf'])
    mjcf=ET.parse(folder/'mujoco/scorpion.xml').getroot()
    checks['mujoco_has_no_pair_exclusions']=not mjcf.findall('contact/exclude') and mjcf.find('option/flag').get('filterparent')=='disable'
    checks['mujoco_collision_count']=len([g for g in mjcf.findall('.//geom') if '_collision_' in g.get('name','')])==expected
    from pxr import Usd, UsdPhysics
    stage=Usd.Stage.Open(str(folder/'isaac_sim/scorpion.usd'))
    collision_prims=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    checks['usd_collision_count']=len(collision_prims)==expected
    checks['usd_no_filtered_pairs']=not any(p.HasAPI(UsdPhysics.FilteredPairsAPI) for p in stage.Traverse())
    checks['usd_joint_collision_enabled']=all(UsdPhysics.Joint(p).GetCollisionEnabledAttr().Get() for p in stage.Traverse() if p.IsA(UsdPhysics.Joint))
    sdf=ET.parse(folder/'gazebo_jetty/models/scorpion/model.sdf').getroot().find('model')
    proxies=[l for l in sdf.findall('link') if l.get('name').startswith('collision_proxy__')]
    checks['sdf_collision_count']=len(sdf.findall('.//collision'))==expected
    checks['sdf_proxy_count']=len(proxies)==len(geometry)
    checks['sdf_proxy_fixed_joints']=all(sdf.find(f"joint[@name='collision_fixed__{link}']/child").text=='collision_proxy__'+link for link in geometry)
    checks['all_checks_passed']=all(checks.values())
    report={'scope':__doc__,'source_urdf_sha256':hashlib.sha256((folder/'genesis/scorpion.urdf').read_bytes()).hexdigest(),
            'collision_pieces':expected,'checks':checks,'isaac_physx_runtime_tested':False,'gazebo_runtime_tested':False}
    (folder/'source/collision_package_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    assert checks['all_checks_passed'],checks
    print(json.dumps(report))

if __name__=='__main__':main()
