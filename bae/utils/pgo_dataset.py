import os
import hashlib
import torch
import numpy as np
import pypose as pp
import torch.utils.data as Data
from urllib.request import urlretrieve

_PGO_DATASETS = {
    "grid3D.g2o": {
        "url": "https://github.com/pypose/bae/releases/download/0.2.2/grid3D.g2o",
        "sha256": "f5dd7a17bf8b9c39890b14f9bb0c1e8366a7490f01714e43e3bc03f52cf414e7",
    },
    "parking-garage.g2o": {
        "url": "https://github.com/pypose/bae/releases/download/0.2.2/parking-garage.g2o",
        "sha256": "3ac0a31bfb601d7455d451e2546655cb5dececf51a7823f57c8a7e0fe1ca6527",
    },
    "sphere_bignoise_vertex3.g2o": {
        "url": "https://github.com/pypose/bae/releases/download/0.2.2/sphere_bignoise_vertex3.g2o",
        "sha256": "484aa1999084d353d83725ba1d992cb709ad3a7e6c396155cc8e87a059c645db",
    },
    "sphere_g2o.g2o": {
        "url": "https://github.com/pypose/bae/releases/download/0.2.2/sphere_g2o.g2o",
        "sha256": "be8dbad53b43695bfa3246add2f92307c3d7340fc5a5641a6f3e46e3e7d0fc61",
    },
    "torus3D.g2o": {
        "url": "https://github.com/pypose/bae/releases/download/0.2.2/torus3D.g2o",
        "sha256": "1ef8831023b372188d33ca63ce0da079d51df89364df71f2bce1c5c7c69ed5ec",
    },
}


def _verify_sha256(filepath, expected_sha256):
    with open(filepath, 'rb') as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual != expected_sha256:
        os.remove(filepath)
        raise RuntimeError(
            f"SHA256 mismatch for {os.path.basename(filepath)}: "
            f"expected {expected_sha256}, got {actual}"
        )


def _download_pgo_dataset(root, dataname):
    if dataname not in _PGO_DATASETS:
        raise ValueError(
            f"Unknown PGO dataset: {dataname}. "
            f"Available: {list(_PGO_DATASETS.keys())}"
        )

    info = _PGO_DATASETS[dataname]
    filepath = os.path.join(root, dataname)

    os.makedirs(root, exist_ok=True)

    if os.path.exists(filepath):
        _verify_sha256(filepath, info["sha256"])
        return

    urlretrieve(info["url"], filepath)
    _verify_sha256(filepath, info["sha256"])


class G2OPGO(Data.Dataset):
    '''
    This data is from the following paper:
    L. Carlone, R. Tron, K. Daniilidis, and F. Dellaert. Initialization Techniques for 3D
    SLAM: a Survey on Rotation Estimation and its Use in Pose Graph Optimization. In IEEE
    Intl. Conf. on Robotics and Automation (ICRA), pages 4597-4604, 2015.
    '''
    def __init__(self, root, dataname, device='cpu', download=True):
        super().__init__()

        if download:
            _download_pgo_dataset(root, dataname)

        def info2mat(info):
            mat = np.zeros((6, 6))
            ix = 0
            for i in range(mat.shape[0]):
                mat[i, i:] = info[ix:ix + (6 - i)]
                mat[i:, i] = info[ix:ix + (6 - i)]
                ix += (6 - i)
            return mat

        self.dtype = torch.get_default_dtype()
        filename = os.path.join(root, dataname)
        ids, nodes, edges, poses, infos = [], [], [], [], []
        with open(filename) as f:
            for line in f:
                line = line.split()
                if line[0] == 'VERTEX_SE3:QUAT':
                    ids.append(torch.tensor(int(line[1]), dtype=torch.int64))
                    nodes.append(pp.SE3(np.array(line[2:], dtype=np.float64)))
                elif line[0] == 'EDGE_SE3:QUAT':
                    edges.append(torch.tensor(np.array(line[1:3], dtype=np.int64)))
                    poses.append(pp.SE3(np.array(line[3:10], dtype=np.float64)))
                    infos.append(torch.tensor(info2mat(np.array(line[10:], dtype=np.float64))))

        self.ids = torch.stack(ids)
        self.nodes = torch.stack(nodes).to(self.dtype).to(device)
        self.edges = torch.stack(edges).to(device)  # have to be LongTensor
        self.poses = torch.stack(poses).to(self.dtype).to(device)
        self.infos = torch.stack(infos).to(self.dtype).to(device)
        assert self.ids.size(0) == self.nodes.size(0) \
               and self.edges.size(0) == self.poses.size(0) == self.infos.size(0)

    def init_value(self):
        return self.nodes.clone()

    def __getitem__(self, i):
        return self.edges[i], self.poses[i], self.infos[i]

    def __len__(self):
        return self.edges.size(0)
