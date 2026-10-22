"""Repair the supplied CAD export in place; no robot assets are vendored in Sim-Agent.

The export mirrored LEFT collision pieces in the wrong coordinate frame and
excluded all adjacent links. Repair their frames, clear the internal rotary
coupling using its measured horn envelope, then enable adjacent contact surfaces.
"""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.robots.urdf import URDFModel, mesh_path, origin
from backend.app.robots.collision_geometry import joint_bore_parts


def xml_write(root, path):
    ET.indent(root); ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--restore-backup',action='store_true',help='Restore only files managed by the previous repair, then rebuild')
    args=parser.parse_args();folder=args.folder.resolve()
    model=URDFModel(folder/'genesis/scorpion.urdf')
    if not args.apply:
        print('Use --apply to back up and repair this local asset package.');return
    backup=folder.parent/'old_no_use'/'for_sim_collision_before_20261021.zip'
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():
        with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in folder.rglob('*'):
                if path.is_file():archive.write(path,path.relative_to(folder))
    elif (folder/'source/collision_repair.json').exists():
        if not args.restore_backup:
            raise SystemExit('Package has already been repaired; use --restore-backup to rebuild managed files')
        managed={'genesis/scorpion.urdf','urdf/scorpion.urdf','urdf/scorpion_ros.urdf','genesis/scorpion.srdf','urdf/scorpion.srdf','mujoco/scorpion.xml','source/collision_config.json','genesis/collision_config.json','urdf/collision_config.json','genesis/run.py'}
        with zipfile.ZipFile(backup) as archive:
            for name in archive.namelist():
                if name in managed or '/meshes/collision/' in name and name.endswith('.obj'):
                    (folder/name).write_bytes(archive.read(name))
        for name in ['source/collision_repair.json','source/collision_repair_geometry.json']:
            (folder/name).unlink(missing_ok=True)
        for path in folder.rglob('*_bore_*.obj'):
            path.unlink()
        model=URDFModel(folder/'genesis/scorpion.urdf')
    meshes={}
    for link,element in model.links.items():
        meshes[link]=[]
        for item in element.findall('collision'):
            filename=item.find('geometry/mesh').get('filename')
            geometry=trimesh.load(mesh_path(filename,model.path),force='mesh',process=True)
            geometry.apply_transform(origin(item.find('origin')))
            meshes[link].append((filename,geometry))
    # Link origins are mirrored across source X. Reflect the CORRESPONDING
    # right local mesh; do not mirror a world-space mesh and subtract another
    # origin. Source visual symmetry was independently checked to <0.3mm.
    reflection=np.diag([-1.,1.,1.,1.])
    for i in range(1,9):
        right,left=f'r_link_{i}',f'l_link_{i}'
        if len(meshes[right])!=len(meshes[left]):raise ValueError('Unexpected asymmetric export')
        meshes[left]=[(old[0],new[1].copy().apply_transform(reflection)) for old,new in zip(meshes[left],meshes[right])]
    report={'backup':str(backup),'adjacent_link_exclusions_removed':True,'bearing_clearance_m':.0002, 'joints':{}, 'links':{}}
    for joint in model.joints:
        parent=joint['parent'];transform=np.asarray(joint['origin'])
        pivot=transform[:3,3];axis=transform[:3,:3]@joint['axis']
        horn_points=np.vstack([g.vertices for name,g in meshes[joint['child']] if 'horn_' in name])
        horn_points=horn_points@transform[:3,:3].T+transform[:3,3]
        delta=horn_points-pivot
        radius=float(np.linalg.norm(delta-np.outer(delta@axis,axis),axis=1).max()+.0002)
        report['joints'][joint['name']]={'bore_radius_m':radius, 'maximum_radial_overcut_m':float(radius*(1/np.cos(np.pi/16)-1))}
        replaced=[]
        for filename,mesh in meshes[parent]:
            if f"joint{joint['name'][5:-2]}_motor_case_{joint['name'][-1]}_" not in Path(filename).name:
                replaced.append((filename,mesh));continue
            pieces=joint_bore_parts(mesh,pivot,axis,radius)
            if not pieces:raise RuntimeError(f"Shaft operation removed the entire case: {joint['name']} {filename}")
            for k,piece in enumerate(pieces):
                replaced.append((str(Path(filename).with_name(Path(filename).stem+f'_bore_{k}.obj')),piece))
        meshes[parent]=replaced
    for link,entries in meshes.items():
        report['links'][link]={'collision_pieces':len(entries)}
        for name,mesh in entries:
            for directory in [folder/'genesis',folder/'urdf',folder/'mujoco',folder/'gazebo_jetty/models/scorpion']:
                path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);mesh.export(path)
    for path in [folder/'genesis/scorpion.urdf',folder/'urdf/scorpion.urdf',folder/'urdf/scorpion_ros.urdf']:
        tree=ET.parse(path);root=tree.getroot()
        for link,entries in meshes.items():
            element=root.find(f"link[@name='{link}']")
            for old in list(element.findall('collision')):element.remove(old)
            for k,(name,_) in enumerate(entries):
                collision=ET.SubElement(element,'collision',name=f'{link}_collision_{k}')
                geometry=ET.SubElement(collision,'geometry')
                # ROS paths retain the package prefix; sibling packages are relative.
                prefix='package://scorpion_description/' if path.stem.endswith('_ros') else ''
                ET.SubElement(geometry,'mesh',filename=prefix+name)
        xml_write(root,path)
    for path in [folder/'genesis/scorpion.srdf',folder/'urdf/scorpion.srdf']:
        root=ET.parse(path).getroot()
        for item in list(root.findall('disable_collisions')):root.remove(item)
        xml_write(root,path)
    path=folder/'mujoco/scorpion.xml';root=ET.parse(path).getroot()
    for old in list(root.find('asset')):
        if '_collision_' in old.get('name',''):root.find('asset').remove(old)
    for link,entries in meshes.items():
        body=root.find(f".//body[@name='{link}']")
        for old in list(body.findall('geom')):
            if '_collision_' in old.get('name',''):body.remove(old)
        for k,(name,_) in enumerate(entries):
            key=f'{link}_collision_{k}';ET.SubElement(root.find('asset'),'mesh',name=key,file=name)
            ET.SubElement(body,'geom',name=key,type='mesh',mesh=key,condim='6')
    for old in list(root.find('contact')):
        if old.tag=='exclude':root.find('contact').remove(old)
    root.find('option/flag').set('filterparent','disable');xml_write(root,path)
    for path in [folder/'source/collision_config.json',folder/'genesis/collision_config.json',folder/'urdf/collision_config.json']:
        config=json.loads(path.read_text());config['excluded_adjacent_pairs']=[];config['adjacent_collision']=True
        config['internal_joint_clearance']='servo shaft bore in collision mesh; no whole-link exclusion'
        path.write_text(json.dumps(config,indent=2)+'\n')
    path=folder/'genesis/run.py';text=path.read_text().replace('enable_adjacent_collision=False','enable_adjacent_collision=True').replace("'adjacent_collision':False","'adjacent_collision':True")
    text=text.replace('import argparse','import argparse\nimport xml.etree.ElementTree as ET')
    text=text.replace('max_collision_pairs=512','max_collision_pairs=max(512, len(ET.parse(Path(__file__).with_name(\'scorpion.urdf\')).findall(\'.//collision\')) * 4)')
    text=text.replace('recompute_inertia=False,','recompute_inertia=False, default_armature=None,')
    path.write_text(text)
    # USD and SDF are updated by a second pass using the same geometry map.
    (folder/'source/collision_repair_geometry.json').write_text(json.dumps({link:[name for name,_ in entries] for link,entries in meshes.items()},indent=2)+'\n')
    (folder/'source/collision_repair.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'collision_pieces':sum(len(x) for x in meshes.values()),'backup':str(backup)}))

if __name__=='__main__':main()
