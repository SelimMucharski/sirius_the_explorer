"""Minimal reader for Unity scene YAML: world poses of transforms and prefab overrides."""

import re

import numpy as np


def _parse_docs(path):
    docs, cur = {}, None
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = re.match(r'^--- !u!(\d+) &(\d+)', line)
            if m:
                cur = {'type': int(m.group(1)), 'id': m.group(2), 'lines': []}
                docs[cur['id']] = cur
            elif cur is not None:
                cur['lines'].append(line.rstrip('\n'))
    for d in docs.values():
        d['text'] = '\n'.join(d['lines'])
    return docs


def _vec(text, key):
    m = re.search(key + r': \{([^}]*)\}', text)
    if not m:
        return None
    return {k.strip(): float(v) for k, v in (p.split(':') for p in m.group(1).split(','))}


def _ref(text, key):
    m = re.search(key + r': \{fileID: (-?\d+)', text)
    return m.group(1) if m else None


def quat_to_mat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class UnityScene:
    """World poses (Unity frame: left-handed, Y up) of every Transform in a .unity file."""

    def __init__(self, path):
        self.docs = _parse_docs(path)
        self._gos = {i: d for i, d in self.docs.items() if d['type'] == 1}
        self._trs = {i: d for i, d in self.docs.items() if d['type'] == 4}

    def prefab_instances(self):
        """[(source_prefab_guid, {target_fileID: {propertyPath: value}})]."""
        out = []
        for d in self.docs.values():
            if d['type'] == 1001:
                out.extend(self.prefab_instances_of(d))
        return out

    def _prefab_mods(self, t):
        """Overrides that target this stripped transform or its GameObject.

        Overrides of other objects inside the prefab (e.g. child transforms) are
        excluded so they cannot shadow the root's values.
        """
        d = self.docs.get(_ref(t['text'], 'm_PrefabInstance'))
        if not d:
            return {}
        own = {_ref(t['text'], 'm_CorrespondingSourceObject')}
        go = self._gos.get(_ref(t['text'], 'm_GameObject'))
        if go:
            own.add(_ref(go['text'], 'm_CorrespondingSourceObject'))
        mods, all_mods = {}, {}
        for src, per_target in self.prefab_instances_of(d):
            for tgt, props in per_target.items():
                all_mods.update(props)
                if tgt in own:
                    mods.update(props)
        # m_Name usually targets the root GameObject, which is not stripped in the scene
        mods.setdefault('m_Name', all_mods.get('m_Name', '?'))
        for k in ('id', 'dict', 'size', 'height', 'latitude', 'longitude', 'altitude'):
            if k in all_mods:
                mods.setdefault(k, all_mods[k])
        return mods

    def prefab_instances_of(self, d):
        src = re.search(r'm_SourcePrefab: \{fileID: \d+, guid: (\w+)', d['text']).group(1)
        mods = {}
        for tgt, prop, val in re.findall(
                r'- target: \{fileID: (-?\d+)[^\n]*\n(?:[^\n]*\n)?\s+propertyPath: (\S+)\n'
                r'\s+value: ?(.*)', d['text']):
            mods.setdefault(tgt, {})[prop] = val
        return [(src, mods)]

    def _parent(self, t):
        par = _ref(t['text'], 'm_Father')
        if par is None:  # stripped transform of a prefab instance
            d = self.docs.get(_ref(t['text'], 'm_PrefabInstance'))
            par = _ref(d['text'], 'm_TransformParent') if d else None
        return par if par in self._trs else None

    def _local(self, t):
        p = _vec(t['text'], 'm_LocalPosition')
        q = _vec(t['text'], 'm_LocalRotation')
        s = _vec(t['text'], 'm_LocalScale')
        if p is None:  # stripped prefab transform: values come from the instance overrides
            m = self._prefab_mods(t)
            p = {a: float(m.get('m_LocalPosition.' + a, 0)) for a in 'xyz'}
            q = {a: float(m.get('m_LocalRotation.' + a, 1 if a == 'w' else 0)) for a in 'xyzw'}
            s = {a: 1.0 for a in 'xyz'}  # root scale overrides are not used in these scenes
        M = np.eye(4)
        M[:3, :3] = quat_to_mat(q['x'], q['y'], q['z'], q['w']) @ np.diag([s['x'], s['y'], s['z']])
        M[:3, 3] = [p['x'], p['y'], p['z']]
        return M

    def _world(self, tid):
        t = self._trs[tid]
        par = self._parent(t)
        return self._world(par) @ self._local(t) if par else self._local(t)

    def name(self, tid):
        t = self._trs[tid]
        go = _ref(t['text'], 'm_GameObject')
        if go in self._gos:
            return re.search(r'm_Name: (.*)', self._gos[go]['text']).group(1)
        return self._prefab_mods(t).get('m_Name', '?')

    def root_prefab_pose(self, name):
        """World matrix of a top-level prefab instance called `name`.

        Needed for instances that nothing references, so the scene stores no
        stripped Transform for them (e.g. an imported FBX terrain).
        """
        found = []
        for d in self.docs.values():
            if d['type'] != 1001 or _ref(d['text'], 'm_TransformParent') not in (None, '0'):
                continue
            for _, per_target in self.prefab_instances_of(d):
                if not any(p.get('m_Name') == name for p in per_target.values()):
                    continue
                for props in per_target.values():
                    if 'm_LocalPosition.x' in props:
                        g = lambda k, dflt: float(props.get(k, dflt))  # noqa: E731
                        M = np.eye(4)
                        M[:3, :3] = quat_to_mat(*(g('m_LocalRotation.' + a, 1 if a == 'w' else 0)
                                                  for a in 'xyzw'))
                        M[:3, 3] = [g('m_LocalPosition.' + a, 0) for a in 'xyz']
                        found.append(M)
        if len(found) != 1:
            raise LookupError(f'Expected one top-level prefab instance {name!r}, found {len(found)}')
        return found[0]

    def objects(self):
        """[(name, 4x4 world matrix, prefab overrides or {})] for every transform."""
        return [(self.name(tid), self._world(tid), self._prefab_mods(t))
                for tid, t in self._trs.items()]
