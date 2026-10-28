"""
Blender (tło): fotoskany kamieni Poly Haven (CC0) -> lekkie OBJ z UV i teksturą do dressingu.

  blender -b -P decimate_rocks.py -- <katalog polyhaven/models/stones> <wyjście> [trójkąty]

Każdy obiekt meshowy z namaqualand_stones_01 i stone_01 jest zmniejszany modyfikatorem
Decimate (collapse, UV zachowane) do ~N trójkątów i zapisywany jako <nazwa>.obj (+ .mtl),
a tekstura diff_1k kopiowana obok.
"""
import os
import shutil
import sys

import bpy

argv = sys.argv[sys.argv.index('--') + 1:]
src, out = argv[0], argv[1]
target = int(argv[2]) if len(argv) > 2 else 800
os.makedirs(out, exist_ok=True)
for name in ('namaqualand_stones_01', 'stone_01'):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.join(src, name, f'{name}_1k.gltf'))
    tex = os.path.join(src, name, 'textures', f'{name}_diff_1k.jpg')
    shutil.copy(tex, os.path.join(out, f'{name}_diff.jpg'))
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    for i, o in enumerate(meshes):
        bpy.ops.object.select_all(action='DESELECT')
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        n = len(o.data.polygons)
        m = o.modifiers.new('dec', 'DECIMATE')
        m.ratio = min(1.0, target / max(n, 1))
        bpy.ops.object.modifier_apply(modifier='dec')
        path = os.path.join(out, f'{name}_{i}.obj')
        bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True, export_uv=True,
                              export_normals=True, export_materials=False,
                              forward_axis='Y', up_axis='Z', apply_modifiers=True)
        print('rock', path, n, '->', len(o.data.polygons))
