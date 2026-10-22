"""Refine only collision proxies that produce contacts outside the actual visual solid.

MuJoCo is used here as an offline geometry inspector of the exported MJCF. The
production state and grasp verification remain owned by Genesis.
"""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.app.robots.urdf import URDFModel,mesh_path
from backend.app.robots.collision_geometry import faithful_convex_parts,float32_convex_piece
from scripts.repair_scorpion_collisions import xml_write


def write_geometry(folder, meshes):
    for link,entries in meshes.items():
        meshes[link]=[(name,clean) for name,mesh in entries if (clean:=float32_convex_piece(mesh)) is not None]
    for link,entries in meshes.items():
        for filename,mesh in entries:
            for directory in ['genesis','urdf','mujoco','gazebo_jetty/models/scorpion']:
                path=folder/directory/filename;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(trimesh.exchange.obj.export_obj(mesh,digits=17))
    for name in ['genesis/scorpion.urdf','urdf/scorpion.urdf','urdf/scorpion_ros.urdf']:
        path=folder/name;root=ET.parse(path).getroot()
        for link,entries in meshes.items():
            body=root.find(f"link[@name='{link}']")
            original=body.find('collision/geometry/mesh').get('filename')
            prefix=original.split('meshes/')[0] if original.startswith('package://') else ''
            for c in list(body.findall('collision')):body.remove(c)
            for i,(filename,_) in enumerate(entries):
                c=ET.SubElement(body,'collision',name=f'{link}_collision_{i}');g=ET.SubElement(c,'geometry')
                ET.SubElement(g,'mesh',filename=prefix+filename)
        xml_write(root,path)
    path=folder/'mujoco/scorpion.xml';root=ET.parse(path).getroot()
    for a in list(root.find('asset')):
        if '_collision_' in a.get('name',''):root.find('asset').remove(a)
    for link,entries in meshes.items():
        body=root.find(f".//body[@name='{link}']")
        for c in list(body.findall('geom')):
            if '_collision_' in c.get('name',''):body.remove(c)
        for i,(filename,_) in enumerate(entries):
            name=f'{link}_collision_{i}';ET.SubElement(root.find('asset'),'mesh',name=name,file=filename)
            ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,condim='6')
    xml_write(root,path)
    (folder/'source/collision_repair_geometry.json').write_text(json.dumps({link:[n for n,_ in entries] for link,entries in meshes.items()},indent=2)+'\n')
    (folder/'source/collision_precision.json').write_text(json.dumps({
        'float32_convex_surfaces':True,'collision_pieces':sum(map(len,meshes.values())),
        'maximum_vertex_quantization_error_m':max((m.metadata.get('vertex_quantization_error_m',0.) for entries in meshes.values() for _,m in entries),default=0.)},indent=2)+'\n')


def audit(folder, model):
    m=mujoco.MjModel.from_xml_path(str(folder/'mujoco/scorpion.xml'));d=mujoco.MjData(m)
    visuals={link:trimesh.load(mesh_path(e.find('visual/geometry/mesh').get('filename'),model.path),force='mesh')
             for link,e in model.links.items()}
    lookup={f'{link}_collision_{i}':(link,i) for link,e in model.links.items() for i,_ in enumerate(e.findall('collision'))}
    candidates={};real=[];worst=0.
    for use_initial in (False,True):
        d.qpos[:]=0.
        initial=json.loads((folder/'genesis/initial_state.json').read_text()) if use_initial else {}
        for j in model.active:
            address=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,j['name'])]
            d.qpos[address]=-.6 if j['name'].startswith('joint7_') else initial.get(j['name'],0.)
        mujoco.mj_forward(m,d)
        for c in d.contact:
            if c.dist>=-.0003 or any(m.geom_bodyid[g]==0 for g in (c.geom1,c.geom2)):continue
            worst=max(worst,float(-c.dist));ghosts=[]
            for order,g in enumerate((c.geom1,c.geom2)):
                key=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g))
                if key not in lookup:continue
                link,index=lookup[key];body=int(m.geom_bodyid[g])
                point=c.pos + (1. if order else -1.)*c.frame[:3]*c.dist/2.
                local=(point-d.xpos[body])@d.xmat[body].reshape(3,3)
                distance=float(trimesh.proximity.signed_distance(visuals[link],local[None,:])[0])
                if distance<-.00025:ghosts.append((-distance,link,index))
            if ghosts:
                error,link,index=max(ghosts)
                # Corresponding left meshes use the independently checked rigid reflection.
                if link.startswith('l_'):link='r_'+link[2:]
                candidates[(link,index)]=max(candidates.get((link,index),0.),error)
            else:real.append({'penetration_m':float(-c.dist),'geoms':[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)]})
    return candidates,real,worst,visuals


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--passes',type=int,default=3)
    args=parser.parse_args();folder=args.folder.resolve();report=[]
    for iteration in range(args.passes):
        model=URDFModel(folder/'genesis/scorpion.urdf')
        candidates,real,worst,visuals=audit(folder,model)
        print('pass',iteration,'ghost proxies',len(candidates),'maximum penetration mm',worst*1000,'contacts within surface tolerance',len(real),flush=True)
        report.append({'pass':iteration,'ghost_proxies':len(candidates),'maximum_penetration_m':worst,'contacts_within_surface_tolerance':real})
        (folder/'source/collision_surface_refinement.json').write_text(json.dumps(report,indent=2)+'\n')
        if not candidates:break
        meshes={link:[(e.find('geometry/mesh').get('filename'),trimesh.load(mesh_path(e.find('geometry/mesh').get('filename'),model.path),force='mesh')) for e in body.findall('collision')]
                for link,body in model.links.items()}
        replacement={}
        for (link,index),error in sorted(candidates.items(),key=lambda item:-item[1]):
            filename,proxy=meshes[link][index]
            tolerance=2e-8/(8**iteration)
            pieces=faithful_convex_parts(proxy,visuals[link],volume_tolerance=tolerance,linear_tolerance=None)
            replacement[(link,index)]=[(str(Path(filename).with_name(Path(filename).stem+f'_faithful{iteration}_{i}.obj')),mesh) for i,(mesh,_) in enumerate(pieces)]
            print(link,index,'surface error mm',round(error*1000,3),'pieces',len(pieces),flush=True)
        for link in meshes:
            if link.startswith('l_'):continue
            entries=[]
            for i,old in enumerate(meshes[link]):entries.extend(replacement.get((link,i),[old]))
            meshes[link]=entries
            if link.startswith('r_'):
                meshes['l_'+link[2:]]=[(name.replace('_R_','_L_'),mesh.copy().apply_transform(np.diag([-1.,1.,1.,1.]))) for name,mesh in entries]
        write_geometry(folder,meshes)
    print('Source collision refinement finished; rerun the audit to verify remaining contacts.',flush=True)

if __name__=='__main__':main()
