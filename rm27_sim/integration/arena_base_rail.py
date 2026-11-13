"""Merge the audited carriage into a repaired arena without dropping terrain repairs."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from .base_rail import read_mesh, sha256, numbers, runtime_hashes
from rm27_sim.simulation.field_collision import _surface_labels


def build(arena, carriage_source, output):
    arena, carriage_source, output = map(lambda p: Path(p).resolve(), (arena, carriage_source, output))
    manifest = json.loads((carriage_source/'assembly_manifest.json').read_text())
    for name, digest in manifest['runtime_hashes'].items():
        if sha256(carriage_source/name) != digest:
            raise ValueError(f'Carriage input changed: {name}')
    source_hashes = runtime_hashes(arena)
    source_hashes['surface_repair_sources.json'] = sha256(arena/'surface_repair_sources.json')
    cfg = manifest['source_config']
    mesh_labels = {}
    support_labels = {}
    for name, rule in cfg['sources'].items():
        if sha256(arena/'assets'/f'{name}.msh') != rule['sha256']:
            raise ValueError(f'Arena visual source changed: {name}')
        v, _, _, f = read_mesh(arena/'assets'/f'{name}.msh')
        labels = mesh_labels[name] = _surface_labels(v, f)
        support_labels[name] = {int(i) for i in np.unique(labels)
                               if v[np.unique(f[labels == i]), 2].min() >= 1.10}
    source = ET.parse(carriage_source/'model.xml')
    model = ET.parse(arena/'model.xml')
    collisions = ET.parse(arena/'collision_geoms.xml')
    collision_assets = ET.parse(arena/'collision_assets.xml')
    repairs = ET.parse(arena/'surface_repair_geoms.xml')
    repair_assets = ET.parse(arena/'surface_repair_assets.xml')
    repair_faces = json.loads((arena/'surface_repair_sources.json').read_text())['patch_source_faces']
    world = model.find('worldbody')
    pivot = np.array(cfg['pivot_m'])
    bodies = {name: copy.deepcopy(source.find(f"worldbody/body[@name='{name}']"))
              for name in ['blue_rail_carriage', 'blue_rail_support']}
    # Keep the original grouping to preserve the internal-contact exclusion.
    for name, body in bodies.items():
        if body is None:
            raise ValueError(f'Missing body {name}')
        world.append(body)
    for tag in ['actuator', 'contact']:
        if model.find(tag) is not None:
            raise ValueError(f'Arena already has {tag}; review integration explicitly')
        model.getroot().append(copy.deepcopy(source.find(tag)))
    for name in manifest['moving_collision_names'] + manifest['support_collision_names']:
        for parent in collisions.iter():
            match = next((g for g in parent if g.tag == 'geom' and g.get('name') == name), None)
            if match is not None:
                parent.remove(match)
                break
        else:
            if name != 'visual_116_detail_0':
                raise ValueError(f'Arena removed a required base collider: {name}')
    old_fixed = {g.get('name'): g for g in ET.parse(carriage_source/'collision_geoms.xml').iter('geom')}
    corrections = []
    for g in collisions.iter('geom'):
        old = old_fixed.get(g.get('name'))
        if old is not None and g.attrib != old.attrib:
            if {k for k in g.attrib.keys() | old.attrib.keys() if g.get(k) != old.get(k)} != {'pos'}:
                raise ValueError(f'Unexpected source collider change: {g.get("name")}')
            delta = np.fromstring(old.get('pos'), sep=' ')-np.fromstring(g.get('pos'), sep=' ')
            if not np.isclose(np.linalg.norm(delta), .0025, atol=1e-8):
                raise ValueError('Unexpected armor offset')
            g.set('pos', old.get('pos'))
            corrections.append(g.get('name'))
    fixed_body = collisions.find('body')
    fixed_body.append(copy.deepcopy(old_fixed['visual_124_detail_0']))
    for mesh in ET.parse(carriage_source/'collision_assets.xml').iter('mesh'):
        if mesh.get('name') in ['visual_116_detail_0', 'visual_124_detail_0']:
            collision_assets.find('asset').append(copy.deepcopy(mesh))
    removed, moving, support = [], [], []
    repair_body = repairs.find('body')
    for g in list(repair_body):
        name = g.get('name')
        index = int(name.rsplit('_', 1)[1])
        faces = repair_faces[index]
        owners = set()
        for mesh, face in faces:
            if mesh in ['visual_116', 'visual_124']:
                owners.add('lens')
            elif mesh in cfg['sources']:
                label = int(mesh_labels[mesh][face])
                if label in cfg['sources'][mesh]['moving_components']:
                    owners.add('moving')
                elif label in support_labels[mesh]:
                    owners.add('support')
                else:
                    owners.add('fixed')
            else:
                owners.add('fixed')
        if len(owners) != 1:
            raise ValueError(f'Repair spans different rigid parts: {name}: {owners}')
        owner = owners.pop()
        if owner == 'lens':
            repair_body.remove(g)
            removed.append(name)
        elif owner == 'moving':
            repair_body.remove(g)
            g.set('pos', numbers(np.fromstring(g.get('pos'), sep=' ')-pivot))
            bodies['blue_rail_carriage'].append(g)
            moving.append(name)
        elif owner == 'support':
            repair_body.remove(g)
            bodies['blue_rail_support'].append(g)
            support.append(name)
    for asset in list(repair_assets.find('asset')):
        if asset.get('name') in removed:
            repair_assets.find('asset').remove(asset)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(arena, output, dirs_exist_ok=True)
    # Only color meshes split by the carriage change. All other terrain assets stay exact.
    for name in ['visual_assets.xml', 'visual_geoms.xml']:
        shutil.copyfile(carriage_source/name, output/name)
    for name, part in manifest['parts'].items():
        if part['fixed_triangles']:
            for suffix in ['', '_carriage']:
                shutil.copyfile(carriage_source/'assets'/f'{name}{suffix}.msh', output/'assets'/f'{name}{suffix}.msh')
    for name, tree in [('model.xml',model),('collision_geoms.xml',collisions),
                       ('collision_assets.xml',collision_assets),('surface_repair_geoms.xml',repairs),
                       ('surface_repair_assets.xml',repair_assets)]:
        ET.indent(tree, space='  ')
        tree.write(output/name, encoding='UTF-8', xml_declaration=True)
    report = dict(source_arena_hashes=source_hashes, carriage_source_hashes=manifest['runtime_hashes'],
                  source_config=cfg, travel_range_m=manifest['travel_range_m'],
                  guide_span_m=manifest['guide_span_m'], carriage_span_m=manifest['carriage_span_m'],
                  axis=manifest['axis'], parts=manifest['parts'],
                  moving_visual_names=manifest['moving_visual_names'],
                  moving_collision_names=manifest['moving_collision_names']+moving,
                  support_collision_names=manifest['support_collision_names']+support,
                  corrected_fixed_armor_names=corrections,
                  superseded_lens_repair_names=removed,
                  moved_repair_names=moving, support_repair_names=support,
                  note='Only lens repair patches are replaced by the audited closed lens hulls. All other repairs and terrain closures/heightfields are retained.')
    report['runtime_hashes'] = runtime_hashes(output)
    (output/'assembly_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arena', type=Path, required=True)
    parser.add_argument('--carriage-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    result=build(args.arena,args.carriage_source,args.output)
    print(json.dumps({key:len(result[key]) for key in ['moved_repair_names','support_repair_names','superseded_lens_repair_names','corrected_fixed_armor_names']},indent=2))

if __name__ == '__main__':
    main()
