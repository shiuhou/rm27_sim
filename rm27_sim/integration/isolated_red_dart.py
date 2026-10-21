"""Derive a static red-dart scene without changing the reviewed v6 source."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from rm27_sim.paths import ROOT, resolve_asset


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asset_path(scene, root, asset):
    compiler = root.find('compiler')
    kind = 'meshdir' if asset.tag == 'mesh' else 'texturedir'
    directory = '' if compiler is None else compiler.get(kind, compiler.get('assetdir', ''))
    return (scene.parent / directory / asset.get('file')).resolve()


def assembly_assets(root, body):
    """Follow only the selected assembly's mesh/material/texture dependencies."""
    assets = {(e.tag, e.get('name')): e for e in root.findall('asset/*')}
    pending = [(kind, e.get(kind)) for e in body.iter()
               for kind in ('mesh', 'material') if e.get(kind)]
    selected = set()
    while pending:
        key = pending.pop()
        if key in selected:
            continue
        element = assets[key]
        selected.add(key)
        if element.get('texture'):
            pending.append(('texture', element.get('texture')))
    return [e for e in root.findall('asset/*') if (e.tag, e.get('name')) in selected]


def check_geoms(source, source_data, result, result_data, names):
    """Compare world transforms, contact settings and effective material color."""
    for name in names:
        i, j = source.geom(name).id, result.geom(name).id
        for attr in ('geom_xpos', 'geom_xmat'):
            np.testing.assert_allclose(getattr(source_data, attr)[i],
                                       getattr(result_data, attr)[j], atol=1e-8, rtol=0)
        for attr in ('geom_type', 'geom_size', 'geom_contype', 'geom_conaffinity',
                     'geom_group', 'geom_friction', 'geom_solref', 'geom_solimp'):
            np.testing.assert_allclose(getattr(source, attr)[i],
                                       getattr(result, attr)[j], atol=1e-8, rtol=0)
        colors = []
        for model, index in ((source, i), (result, j)):
            material = model.geom_matid[index]
            colors.append(model.mat_rgba[material] if material >= 0 else model.geom_rgba[index])
        np.testing.assert_array_equal(*colors)


def build(output, config=ROOT / 'configs/isolated_red_dart.json'):
    output, config = Path(output).resolve(), Path(config).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    cfg = json.loads(config.read_text())
    scene = (resolve_asset(cfg['source_scene'])).resolve()
    field = (resolve_asset(cfg['field_scene'])).resolve()
    inputs = {str(config): sha256(config), str(scene): sha256(scene), str(field): sha256(field)}
    for path, key in ((scene, 'source_scene_sha256'), (field, 'field_scene_sha256')):
        if inputs[str(path)] != cfg[key]:
            raise ValueError(f'Source changed; review configuration before rebuilding: {path}')
    original_root, field_root = ET.parse(scene).getroot(), ET.parse(field).getroot()
    root = copy.deepcopy(original_root)
    world, assets = root.find('worldbody'), root.find('asset')
    station = field_root.find(f"worldbody/body[@name='{cfg['launcher_body']}']")
    if station is None:
        raise ValueError('Expected a named, world-level red launch station')
    selected_assets = assembly_assets(field_root, station)
    original_assets = original_root.findall('asset/*')
    for source, tree, elements in ((scene, original_root, original_assets),
                                   (field, field_root, selected_assets)):
        for element in elements:
            if element.get('file'):
                path = asset_path(source, tree, element)
                inputs[str(path)] = sha256(path)

    removed = []
    for parent in world.iter():
        for geom in list(parent.findall('geom')):
            name = geom.get('name', '')
            if (name.startswith(tuple(cfg['remove_geom_prefixes']))
                    or name in cfg['remove_geom_names']):
                removed.append(name)
                parent.remove(geom)
    if not removed:
        raise ValueError('No red hangar geometry found in source scene')
    imported = copy.deepcopy(station)
    imported_names = [g.get('name') for g in imported.findall('.//geom')]
    retained_names = [g.get('name') for g in world.findall('.//geom')]
    if not imported_names or set(imported_names) & set(retained_names):
        raise ValueError('Launch station is empty or conflicts with retained geometry')
    # Namespace imported default classes so the retained v6 scene cannot inherit them.
    defaults = root.find('default')
    if defaults is None:
        defaults = ET.Element('default')
        root.insert(list(root).index(assets), defaults)
    for class_name in sorted({e.get('class') for e in imported.iter() if e.get('class')}):
        source_default = field_root.find(f"default/default[@class='{class_name}']")
        if source_default is None:
            raise ValueError(f'Missing launch-station default: {class_name}')
        cloned = copy.deepcopy(source_default)
        renamed = f'red_dart_{class_name}'
        cloned.set('class', renamed)
        defaults.append(cloned)
        for element in imported.iter():
            if element.get('class') == class_name:
                element.set('class', renamed)
    existing_assets = {(e.tag, e.get('name')) for e in assets}
    if any((e.tag, e.get('name')) in existing_assets for e in selected_assets):
        raise ValueError('Imported asset names conflict with retained assets')

    output.mkdir(parents=True)
    (output / 'assets').mkdir()
    report = dict(passed=False, mode='headless static model validation', configuration=cfg,
                  input_sha256=inputs, builder_sha256=sha256(__file__), removed_geoms=removed)
    summary = output / 'summary.json'
    summary.write_text(json.dumps(report, indent=2) + '\n')

    def copy_asset(source, tree, element, prefix):
        cloned = copy.deepcopy(element)
        if element.get('file'):
            path = asset_path(source, tree, element)
            relative = Path('assets') / f'{prefix}_{element.tag}_{element.get("name")}{path.suffix}'
            shutil.copyfile(path, output / relative)
            if sha256(output / relative) != inputs[str(path)]:
                raise ValueError(f'Asset changed during copy: {path}')
            cloned.set('file', relative.as_posix())
        return cloned

    for element in list(assets):
        assets.remove(element)
    for element in original_assets:
        assets.append(copy_asset(scene, original_root, element, 'v6'))
    for element in selected_assets:
        assets.append(copy_asset(field, field_root, element, 'dart'))
    compiler = root.find('compiler')
    for key in ('assetdir', 'meshdir', 'texturedir'):
        compiler.attrib.pop(key, None)
    world.append(imported)
    root.set('model', 'rm27_isolated_blue_base_red_dart')
    statistic = root.find('statistic')
    statistic.set('center', '0.3 -3.1 0.8')
    statistic.set('extent', '28')
    ET.indent(root)
    scene_out = output / 'scene.xml'
    ET.ElementTree(root).write(scene_out, encoding='unicode')

    original = mujoco.MjModel.from_xml_path(str(scene))
    field_model = mujoco.MjModel.from_xml_path(str(field))
    result = mujoco.MjModel.from_xml_path(str(scene_out))
    original_data, field_data, state = [mujoco.MjData(m) for m in (original, field_model, result)]
    for model, data in ((original, original_data), (field_model, field_data), (result, state)):
        mujoco.mj_forward(model, data)
    if result.nq != 0 or result.nu != 0:
        raise ValueError('Expected a static scene with no aircraft or controller')
    for _ in range(100):
        mujoco.mj_step(result, state)
    if not np.isfinite(state.geom_xpos).all():
        raise ValueError('Non-finite static model state')
    check_geoms(original, original_data, result, state, retained_names)
    check_geoms(field_model, field_data, result, state, imported_names)
    for i in range(original.nsite):
        j = result.site(original.site(i).name).id
        for attr in ('site_xpos', 'site_xmat'):
            np.testing.assert_allclose(getattr(original_data, attr)[i],
                                       getattr(state, attr)[j], rtol=0, atol=1e-8)
    if result.ngeom != original.ngeom - len(removed) + len(imported_names):
        raise ValueError('Unexpected added or removed geometry')
    for name in removed:
        if mujoco.mj_name2id(result, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0:
            raise ValueError(f'Removed geometry still exists: {name}')
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f'Input changed during build: {path}')
    station_id = result.body(cfg['launcher_body']).id
    imported_ids = [result.geom(name).id for name in imported_names]
    report.update(
        passed=True, retained_geoms=retained_names, imported_geoms=imported_names,
        geoms=result.ngeom, static_steps=100,
        colliders=int(np.count_nonzero(result.geom_contype | result.geom_conaffinity)),
        station_position_m=state.xpos[station_id].tolist(),
        station_colliders=int(np.count_nonzero(
            result.geom_contype[imported_ids] | result.geom_conaffinity[imported_ids])),
        output_sha256={str(p.relative_to(output)): sha256(p)
                       for p in [scene_out, *sorted((output / 'assets').iterdir())]},
        limitations=[
            'Blue-base tower remains visual-only; its existing provisional armor patch collides.',
            'Red dart station reuses the current field major-panel collision shells; fine hardware remains visual-only.',
            'Separate rm27_dock and slot sites retain the v6 world poses; the dock is static and elevated after pedestal removal.',
            'No aircraft, projectile, flight controller, firing simulation or full-field mission validation.',
        ],
    )
    summary.write_text(json.dumps(report, indent=2) + '\n')
    print(scene_out)
    return report
