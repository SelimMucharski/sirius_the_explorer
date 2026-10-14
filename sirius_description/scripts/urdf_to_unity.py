#!/usr/bin/env python3
"""Export a URDF into a JSON model + STL meshes consumed by the Unity Sirius builder.

All poses are converted from ROS (x fwd, y left, z up, right-handed) into
Unity (x right, y up, z fwd, left-handed): p_u = M p_r, R_u = M R_r M^T.

Usage: urdf_to_unity.py <robot.urdf> <output_dir>
"""

import json
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
from ament_index_python.packages import get_package_share_directory

M = np.array([[0.0, -1.0, 0.0],
              [0.0, 0.0, 1.0],
              [1.0, 0.0, 0.0]])

# Links whose collision geometry is replaced by WheelColliders or ignored in Unity.
SKIP_COLLISION_SUFFIXES = ('wheel', 'rim', 'suspension', 'camera', 'gnss', 'indicator_light')


def rpy_to_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_quat(m):
    """Standard rotation matrix -> quaternion (x, y, z, w); Unity uses the same algebra."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [x, y, z, w]


def vec(text, default='0 0 0'):
    return [float(v) for v in (text or default).split()]


def convert_origin(el):
    xyz = vec(el.get('xyz')) if el is not None else [0, 0, 0]
    rpy = vec(el.get('rpy')) if el is not None else [0, 0, 0]
    pos = M @ np.array(xyz)
    rot = M @ rpy_to_matrix(*rpy) @ M.T
    return {'position': pos.tolist(), 'rotation': matrix_to_quat(rot)}


def resolve_uri(uri):
    if uri.startswith('package://'):
        package, rel = uri[len('package://'):].split('/', 1)
        return os.path.join(get_package_share_directory(package), rel)
    if uri.startswith('file://'):
        return uri[len('file://'):]
    return uri


def convert_geometry(geom, materials, material_el, out_mesh_dir, meshes):
    g = {}
    child = list(geom)[0]
    if child.tag == 'mesh':
        src = resolve_uri(child.get('filename'))
        name = os.path.splitext(os.path.basename(src))[0]
        if name not in meshes:
            shutil.copy(src, os.path.join(out_mesh_dir, name + '.stl'))
            meshes.add(name)
        g = {'type': 'mesh', 'mesh': name, 'scale': vec(child.get('scale'), '1 1 1')}
    elif child.tag == 'box':
        sx, sy, sz = vec(child.get('size'))
        g = {'type': 'box', 'size': [sy, sz, sx]}  # Unity (x=ROS y, y=ROS z, z=ROS x)
    elif child.tag == 'cylinder':
        # ROS cylinder axis = z, which maps to Unity y; Unity cylinder axis is y.
        g = {'type': 'cylinder', 'radius': float(child.get('radius')), 'length': float(child.get('length'))}
    elif child.tag == 'sphere':
        g = {'type': 'sphere', 'radius': float(child.get('radius'))}
    color = [0.7, 0.7, 0.7, 1.0]
    if material_el is not None:
        c = material_el.find('color')
        if c is not None:
            color = vec(c.get('rgba'))
        elif material_el.get('name') in materials:
            color = materials[material_el.get('name')]
    g['color'] = color
    return g


def main():
    urdf_path, out_dir = sys.argv[1], sys.argv[2]
    out_mesh_dir = os.path.join(out_dir, 'StlSource~')
    os.makedirs(out_mesh_dir, exist_ok=True)

    root = ET.parse(urdf_path).getroot()
    materials = {}
    for m in root.findall('material'):
        c = m.find('color')
        if c is not None:
            materials[m.get('name')] = vec(c.get('rgba'))

    meshes = set()
    links = []
    for link in root.findall('link'):
        entry = {'name': link.get('name'), 'visuals': [], 'collisions': []}
        for vis in link.findall('visual'):
            g = convert_geometry(vis.find('geometry'), materials, vis.find('material'), out_mesh_dir, meshes)
            g.update(convert_origin(vis.find('origin')))
            entry['visuals'].append(g)
        if not link.get('name').endswith(SKIP_COLLISION_SUFFIXES):
            for col in link.findall('collision'):
                g = convert_geometry(col.find('geometry'), materials, None, out_mesh_dir, meshes)
                g.update(convert_origin(col.find('origin')))
                entry['collisions'].append(g)
        inertial = link.find('inertial')
        if inertial is not None and inertial.find('mass') is not None:
            entry['mass'] = float(inertial.find('mass').get('value'))
        links.append(entry)

    joints = []
    for j in root.findall('joint'):
        axis = j.find('axis')
        limit = j.find('limit')
        limits = {}
        if limit is not None:
            limits = {k: float(limit.get(k)) for k in ('lower', 'upper', 'velocity') if limit.get(k) is not None}
        joints.append({
            'limit': limits,
            'name': j.get('name'),
            'type': j.get('type'),
            'parent': j.find('parent').get('link'),
            'child': j.find('child').get('link'),
            'axis': (M @ np.array(vec(axis.get('xyz') if axis is not None else '1 0 0'))).tolist(),
            **convert_origin(j.find('origin')),
        })

    model = {'robot': root.get('name'), 'root_link': 'base_link', 'links': links, 'joints': joints}
    with open(os.path.join(out_dir, 'sirius_model.json'), 'w') as f:
        json.dump(model, f, indent=1)
    print(f'Wrote {len(links)} links, {len(joints)} joints, {len(meshes)} meshes to {out_dir}')


if __name__ == '__main__':
    main()
