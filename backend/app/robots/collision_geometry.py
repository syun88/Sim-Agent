"""Collision-only geometry operations; visual meshes and inertias stay authoritative."""
import numpy as np
import trimesh


def float32_convex_piece(mesh):
    """Rebuild on the coordinates used by physics, removing collapsed faces.

    Halfspace intersections can leave nanometre slivers that collapse on float32
    import. Re-hulling quantized vertices keeps the convex solid and produces a
    closed surface without zero-area triangles. A zero-volume piece is omitted.
    """
    original = np.asarray(mesh.vertices, float)
    if not np.isfinite(original).all():
        raise ValueError('Non-finite collision vertices')
    for decimals in (7, 6, 5, 4):
        quantized = np.round(original, decimals).astype(np.float32).astype(float)
        points = np.unique(quantized, axis=0)
        if len(points)<4 or np.linalg.matrix_rank(points-points.mean(axis=0), tol=1e-9)<3:
            return None
        result = trimesh.convex.convex_hull(points)
        if result.volume<1e-13:
            return None
        a, b = result.triangles[:, 1]-result.triangles[:, 0], result.triangles[:, 2]-result.triangles[:, 0]
        sine = np.linalg.norm(np.cross(a, b), axis=1)/np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-30)
        if result.is_watertight and np.all(result.area_faces>0.) and np.all(sine>1e-6):
            result.metadata['vertex_quantization_error_m'] = float(np.linalg.norm(quantized-original, axis=1).max())
            return result
    raise ValueError('Cannot create a stable convex collision surface within 0.1mm lattice precision')


def inset_authored_boundary(mesh, visual, distance):
    """Inset local material faces, including cavity walls, not BSP interfaces.

    Each clipping plane comes from a nearby, aligned authored triangle. Internal
    partition faces do not become new erosion surfaces. Returns None if the
    piece lies entirely in the requested surface skin.
    """
    import manifold3d as mf
    hull = mesh.convex_hull
    closest, separation, triangles = trimesh.proximity.closest_point(visual, hull.triangles_center)
    normals = visual.face_normals[triangles]
    aligned = np.einsum('ij,ij->i', normals, hull.face_normals) > .9
    selected = aligned & (separation < distance+.0001)
    solid = mf.Manifold.hull_points(np.asarray(hull.vertices, float))
    for normal, point in zip(normals[selected], closest[selected]):
        solid = solid.trim_by_plane((-normal).tolist(), float(distance-point@normal))
        if solid.is_empty() or solid.volume()<1e-13:
            return None
    data = solid.to_mesh64()
    return trimesh.Trimesh(np.asarray(data.vert_properties)[:, :3], data.tri_verts, process=True)


def joint_bore_parts(mesh, pivot, axis, radius, sectors=16):
    """Cut a shaft clearance from a convex case without disabling its collisions.

    Each output is a convex intersection of three halfspaces and the original
    case. The bore is a circumscribed polygon; its maximum radial overcut is
    radius * (sec(pi/sectors) - 1). Use only for a known internal rotary bearing,
    never to clear an arbitrary collision along an arm's path.
    """
    import manifold3d as manifold
    hull = mesh.convex_hull
    solid = manifold.Manifold(manifold.Mesh(np.asarray(hull.vertices, np.float32), np.asarray(hull.faces, np.uint32)))
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    pivot = np.asarray(pivot, float)
    basis = np.eye(3)[np.argmin(abs(axis))]
    u = np.cross(axis, basis); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    pieces = []
    for theta in np.linspace(0., 2.*np.pi, sectors, endpoint=False):
        half = np.pi / sectors
        radial = np.cos(theta)*u + np.sin(theta)*v
        left = -np.sin(theta-half)*u + np.cos(theta-half)*v
        right = np.sin(theta+half)*u - np.cos(theta+half)*v
        part = solid
        for normal, offset in [(left, left@pivot), (right, right@pivot), (radial, radial@pivot + radius)]:
            part = part.trim_by_plane(normal.tolist(), float(offset))
        if part.is_empty() or part.volume() < 1e-12:
            continue
        data = part.to_mesh()
        result = trimesh.Trimesh(np.asarray(data.vert_properties)[:,:3], np.asarray(data.tri_verts), process=True)
        pieces.append(result.convex_hull)
    return pieces


def surface_error(collision, visual, sample_count=1000):
    """Surface discrepancy, explicitly not proof of collision or contact success."""
    points = trimesh.sample.sample_surface(collision, sample_count, seed=43)[0]
    distances = trimesh.proximity.closest_point(visual, points)[1]
    return {'median_m': float(np.median(distances)), 'p95_m': float(np.quantile(distances,.95)),
            'maximum_m': float(distances.max())}


def faithful_convex_parts(proxy, visual, volume_tolerance=5e-10, max_depth=18, linear_tolerance=.0003):
    """Intersect an approximate proxy with its own visual solid and refine voids.

    Splits follow the missing material, rather than inflating every through-hole
    into a hull. Each returned piece records the remaining hull excess volume.
    This repairs collision geometry only; it never moves the visual robot.
    """
    import manifold3d as mf
    def manifold(mesh, convex=False):
        if convex:
            return mf.Manifold.hull_points(np.asarray(mesh.vertices,float))
        result=mf.Manifold(mf.Mesh64(np.asarray(mesh.vertices,np.float64),np.asarray(mesh.faces,np.uint64)))
        if result.status()!=mf.Error.NoError:
            raise ValueError('Collision refinement requires an oriented manifold visual mesh')
        return result
    base=manifold(proxy.convex_hull,True)^manifold(visual)
    base_data=base.to_mesh64()
    base_surface=trimesh.Trimesh(np.asarray(base_data.vert_properties)[:,:3],base_data.tri_verts,process=False)
    result=[]
    def outer_error(extra):
        points=[]
        for gap in extra.decompose():
            if gap.volume()<1e-13:continue
            data=gap.to_mesh64()
            vertices=np.asarray(data.vert_properties)[:,:3]
            centers=vertices[np.asarray(data.tri_verts)].mean(axis=1)
            surface=np.vstack([vertices,centers])
            if len(surface)>32:surface=surface[np.linspace(0,len(surface)-1,32,dtype=int)]
            points.extend(surface)
            points.append(np.asarray(gap.bounding_box()).reshape(2,3).mean(axis=0))
        if not points:return 0.
        return float(trimesh.proximity.closest_point(base_surface,np.asarray(points))[1].max())
    def visit(solid,depth):
        for part in solid.decompose():
            if part.volume()<1e-10:continue
            part=part.simplify(.00001)
            native_hull=part.hull()
            data=native_hull.to_mesh64()
            hull=trimesh.Trimesh(np.asarray(data.vert_properties)[:,:3],data.tri_verts,process=False)
            extra=native_hull-base
            excess=max(0.,extra.volume())
            if excess<=volume_tolerance or linear_tolerance is not None and outer_error(extra)<=linear_tolerance:
                result.append((hull,excess));continue
            if depth>=max_depth:
                raise ValueError(f'Convex refinement exceeded its depth limit; excess={excess}, depth={depth}')
            # Slice along the largest extent of the largest missing region.
            gap=max(extra.decompose(),key=lambda x:x.volume())
            bounds=np.asarray(gap.bounding_box()).reshape(2,3)
            axis=int(np.argmax(bounds[1]-bounds[0]))
            midpoint=float(bounds[:,axis].mean())
            own=np.asarray(part.bounding_box()).reshape(2,3)
            if midpoint<=own[0,axis]+1e-6 or midpoint>=own[1,axis]-1e-6:
                axis=int(np.argmax(own[1]-own[0]));midpoint=float(own[:,axis].mean())
            normal=np.eye(3)[axis]
            visit(part.trim_by_plane(normal.tolist(),midpoint),depth+1)
            visit(part.trim_by_plane((-normal).tolist(),-midpoint),depth+1)
    visit(base,0)
    # BSP cells can share a flat face even when the visual contains many small
    # triangles. Merge those cells while checking the complete visual solid;
    # each merged hull is checked against the original clipped solid again.
    if len(result)<200:
        return result
    import heapq
    from rtree import index
    properties=index.Property();properties.dimension=3
    tree=index.Index(properties=properties)
    active={i:mesh for i,(mesh,_) in enumerate(result)}
    bounds={i:mesh.bounds.reshape(-1) for i,mesh in active.items()}
    for i,box in bounds.items():tree.insert(i,box)
    pending=[]
    def propose(a,b):
        if a==b:return
        first,second=active[a],active[b]
        hull=trimesh.convex.convex_hull(np.vstack([first.vertices,second.vertices]))
        excess=hull.volume-first.volume-second.volume
        if excess<=max(volume_tolerance*2.1, (linear_tolerance or 0.)*hull.area):
            heapq.heappush(pending,(float(excess),a,b,hull))
    for a in list(active):
        box=bounds[a].copy();box[:3]-=1e-7;box[3:]+=1e-7
        for b in tree.intersection(box):
            if a<b:propose(a,b)
    next_id=len(active)
    while pending:
        _,a,b,hull=heapq.heappop(pending)
        if a not in active or b not in active:continue
        missing=manifold(hull,True)-base
        extra=max(0.,missing.volume())
        if extra>volume_tolerance and (linear_tolerance is None or outer_error(missing)>linear_tolerance):continue
        for old in (a,b):tree.delete(old,bounds[old]);del active[old]
        i=next_id;next_id+=1;active[i]=hull;bounds[i]=hull.bounds.reshape(-1)
        box=bounds[i].copy();box[:3]-=1e-7;box[3:]+=1e-7
        neighbors=list(tree.intersection(box));tree.insert(i,bounds[i])
        for b in neighbors:propose(i,b)
    return [(mesh,max(0.,(manifold(mesh,True)-base).volume())) for mesh in active.values()]


def clean_solid(mesh):
    """A manifold, consistently wound solid for decomposition."""
    solid = mesh.copy()
    solid.merge_vertices()
    solid.update_faces(solid.nondegenerate_faces())
    return solid


def inset_convex_pieces(urdf_path, *, source_mesh_dir, proxy_dir, proxy_uri_prefix, keep_authored=(),
                        concavity=.01, skin_mm=.7, max_pieces=64):
    """Replace a prepared URDF's collision meshes with inset convex pieces, in place.

    The source meshes are visual-quality parts: neighbouring links overlap by tens of millimetres
    at every joint, and a URDF cannot filter individual pairs. One whole-link convex hull cannot
    fix that either (hulling a concave link adds volume; link3 of one arm grew 4.3x), so CoACD
    (MIT, SIGGRAPH 2022) splits each authored mesh into collision-aware convex pieces and each
    piece is clipped against the authored visual's inset halfspaces by this module's own primitive
    - the piece-level shape of a prepared package. ``keep_authored`` names links whose authored
    collision geometry is the point of them (distal jaws: skinning them would move the aperture and
    the cube size the gripper inference measured).

    Returns the record of what was written; the caller stores it beside the package.
    """
    import copy
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import numpy as np
    import trimesh
    import coacd
    from backend.app.robots.urdf import URDFModel, mesh_path

    tree = ET.parse(urdf_path)
    model = URDFModel(urdf_path)
    source_mesh_dir, proxy_dir = Path(source_mesh_dir), Path(proxy_dir)
    proxy_dir.mkdir(parents=True, exist_ok=True)
    distance = skin_mm / 1000.
    record, pieces_total = {}, 0
    for link, body in model.links.items():
        if link in set(keep_authored) or not body.findall("collision"):
            continue
        visual_element = body.find("visual/geometry/mesh")
        visual = trimesh.load(mesh_path(visual_element.get("filename"), model.path), force="mesh", process=False)
        pieces, authored_volume = [], 0.
        for collision in body.findall("collision"):
            mesh_element = collision.find("geometry/mesh")
            authored = source_mesh_dir / Path(mesh_element.get("filename")).name
            solid = clean_solid(trimesh.load(authored, force="mesh", process=False))
            authored_volume += float(solid.volume)
            decomposed = coacd.run_coacd(
                coacd.Mesh(np.asarray(solid.vertices, np.float64), np.asarray(solid.faces, np.int32)),
                threshold=concavity, max_convex_hull=max_pieces, preprocess_mode="auto",
                preprocess_resolution=50, resolution=1000, mcts_iterations=60, seed=0)
            kept = 0
            for vertices, faces in decomposed:
                piece = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces, np.int64), process=True)
                inset = inset_authored_boundary(piece, visual, distance)
                if inset is None or not len(inset.faces):
                    continue
                inset.export(proxy_dir / f"{link}_{kept}.stl")
                pieces.append(proxy_dir / f"{link}_{kept}.stl")
                kept += 1
            if not kept:
                raise SystemExit(f"{link}: no collision piece survived a {skin_mm:g} mm skin")
        element = next(e for e in tree.getroot().findall("link") if e.get("name") == link)
        # The authored collision carries the mm->m scale and a per-collision origin; a proxy
        # written without them is a thousand times too large and offset (measured: the
        # declaration then read a 16 m "overlap"). Clone it and swap only the mesh.
        template = element.find("collision")
        for existing in element.findall("collision"):
            element.remove(existing)
        for path in pieces:
            collision = copy.deepcopy(template)
            collision.find("geometry/mesh").set("filename", f"{proxy_uri_prefix}/{path.name}")
            element.append(collision)
        pieces_total += len(pieces)
        record[link] = {"pieces": [path.name for path in pieces], "skin_mm": skin_mm,
                        "concavity_threshold": concavity,
                        "volume_authored_mm3": round(authored_volume, 1),
                        "volume_proxy_mm3": round(sum(float(trimesh.load(path, force="mesh").volume) for path in pieces), 1)}
    ET.indent(tree)
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return {"method": "CoACD (MIT) convex pieces, each clipped by inset_authored_boundary",
            "skin_mm": skin_mm, "concavity_threshold": concavity, "max_pieces": max_pieces,
            "keep_authored": sorted(set(keep_authored)), "pieces": pieces_total, "links": record}
