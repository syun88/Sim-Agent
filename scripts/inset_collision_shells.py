"""Apply bounded collision skin to the supplied CAD export, preserving finger pads.

Clip each refined piece against its ORIGINAL convex proxy's inset halfspaces.
Internal subdivision boundaries stay intact. Motor cases use 0.65mm, other arm
surfaces 0.25mm, and distal finger surfaces remain unchanged. This gives the CAD
assembly a small contact clearance without excluding connected links. Visual
meshes, masses and joint frames are unchanged. The original package is retained.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import trimesh
import manifold3d as mf
from scipy.spatial import ConvexHull
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.app.robots.urdf import URDFModel,mesh_path
from scripts.refine_robot_collision_surfaces import write_geometry
from backend.app.robots.collision_geometry import inset_authored_boundary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    parser.add_argument('--boundary-only',action='store_true',help='Complete cavity-wall skin on an already inset package')
    args=parser.parse_args();folder=args.folder;u=URDFModel(folder/'genesis/scorpion.urdf')
    backup=folder.parent/'old_no_use/for_sim_collision_before_20261021.zip'
    meshes={};stats=[];planes={}
    with zipfile.ZipFile(backup) as archive:
        import xml.etree.ElementTree as ET
        root=ET.fromstring(archive.read('genesis/scorpion.urdf'))
        originals={e.find('geometry/mesh').get('filename') for body in root.findall('link') for e in body.findall('collision')}
        for link,body in u.links.items():
            entries=[]
            visual=trimesh.load(mesh_path(body.find('visual/geometry/mesh').get('filename'),u.path),force='mesh')
            for c in body.findall('collision'):
                name=c.find('geometry/mesh').get('filename')
                if args.boundary_only:
                    part=trimesh.load(folder/'genesis'/name,force='mesh')
                    if 'motor_case_' in name:
                        part=inset_authored_boundary(part,visual,.00065)
                    if part is not None:entries.append((name,part))
                    continue
                # Restore the untrimmed refined piece; the new skin is absolute
                # relative to the backup, not cumulative on prior experiments.
                source_name=name.replace('_interface','').replace('_skin','')
                part=trimesh.load(folder/'genesis'/source_name,force='mesh')
                base_name=next(n for n in originals if Path(source_name).stem==Path(n).stem or Path(source_name).stem.startswith(Path(n).stem+'_'))
                case='motor_case_' in base_name
                margin=.00065 if case else (.0 if link.endswith(('_7','_8')) else .00025)
                if case:
                    part=inset_authored_boundary(part,visual,margin)
                    if part is None:continue
                if margin==0.:
                    entries.append((source_name,part));continue
                key=(base_name,margin)
                if key not in planes:
                    reference=base_name.replace('_L_','_R_') if link.startswith('l_') else base_name
                    proxy=trimesh.load(io.BytesIO(archive.read('genesis/'+reference)),file_type='obj',force='mesh')
                    if link.startswith('l_'):proxy.apply_transform(np.diag([-1.,1.,1.,1.]))
                    equations=ConvexHull(proxy.vertices).equations
                    _,unique=np.unique(np.round(equations,8),axis=0,return_index=True)
                    planes[key]=equations[unique]
                solid=mf.Manifold.hull_points(np.asarray(part.vertices,float));before=solid.volume()
                equations=planes[key]
                outside=(part.vertices@equations[:,:3].T+equations[:,3]+margin).max(axis=0)>1e-10
                for equation in equations[outside]:
                    normal,offset=equation[:3],equation[3]
                    solid=solid.trim_by_plane((-normal).tolist(),float(offset+margin))
                    if solid.is_empty():break
                if solid.is_empty() or solid.volume()<1e-13:continue
                data=solid.to_mesh64();result=trimesh.Trimesh(np.asarray(data.vert_properties)[:,:3],data.tri_verts,process=True)
                filename=str(Path(source_name).with_name(Path(source_name).stem+'_skin.obj'))
                entries.append((filename,result));stats.append({'source':source_name,'skin_m':margin,'removed_volume_m3':before-solid.volume()})
            meshes[link]=entries
            print(link,len(entries),'pieces',flush=True)
    write_geometry(folder,meshes)
    report={'policy':__doc__,'backup':str(backup),'collision_pieces':sum(map(len,meshes.values())),
            'finger_surface_skin_m':0.,'pieces':stats}
    if args.boundary_only:
        report['previous_pass']=json.loads((folder/'source/collision_skin.json').read_text())
        report['cavity_boundaries_inset']=True
    (folder/'source/collision_skin.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['pieces','policy','previous_pass']}))

if __name__=='__main__':main()
