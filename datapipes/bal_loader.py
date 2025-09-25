"""
This file contains the pipeline for the Bundle Adjustment in the Large dataset.

The dataset is from the following paper:
Sameer Agarwal, Noah Snavely, Steven M. Seitz, and Richard Szeliski.
Bundle adjustment in the large.
In European Conference on Computer Vision (ECCV), 2010.

Link to the dataset: https://grail.cs.washington.edu/projects/bal/
"""

import torch, os, warnings
import numpy as np
from functools import partial
from operator import itemgetter, methodcaller
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from torchvision.transforms import Compose
from scipy.spatial.transform import Rotation
from torchdata.datapipes.iter import HttpReader, IterableWrapper, FileOpener
import pypose as pp

DTYPE = torch.float64

# ignore bs4 warning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# only export __all__
__ALL__ = ['build_pipeline', 'read_bal_data', 'DATA_URL', 'ALL_DATASETS']

# base url for the BAL dataset, used to download the problem files
DATA_URL = 'https://grail.cs.washington.edu/projects/bal/'

# all dataset names in the BAL dataset, used to check if the dataset name is valid
ALL_DATASETS = ['ladybug', 'trafalgar', 'dubrovnik', 'venice', 'final']

# helper for torchdata, add base url to the file name
_with_base_url = partial(os.path.join, DATA_URL)

# helper for torchdata, check if s ends with b
def _endswith(s, b):
    return s.endswith(b)

# helper for torchdata, check if s is not None
def _not_none(s):
    return s is not None

# extract problem file urls from the problem url
def _problem_lister(*problem_url, cache_dir):
    problem_list_dp = IterableWrapper(problem_url).on_disk_cache(
        filepath_fn=Compose([os.path.basename, partial(os.path.join, cache_dir)]),
    )
    problem_list_dp = HttpReader(problem_list_dp).end_caching(same_filepath_fn=True)

    # read the cached problem list html file
    problem_list_dp = FileOpener(problem_list_dp)
    problem_list_dp = problem_list_dp.readlines(return_path=False
    # parse HTML <a> tag's href attributes using bs4
    ).map(partial(BeautifulSoup, features="html.parser")).map(methodcaller('find', 'a')
    # must end with .bz2
    ).filter(_not_none).map(methodcaller('get', 'href')).filter(partial(_endswith, b='.bz2')
    # add base url
    ).map(_with_base_url)

    # sort the problem files by the number of images
    problem_list_sorted = sorted(list(problem_list_dp), key=lambda x: int(os.path.basename(x).split('-')[1]))
    problem_list_dp = IterableWrapper(problem_list_sorted)

    return problem_list_dp

# download and decompress the problem files
def _download_pipe(cache_dir, url_dp, suffix: str):
    # cache compressed files
    cache_compressed = url_dp.on_disk_cache(
        filepath_fn=Compose([os.path.basename, partial(os.path.join, cache_dir)]) ,
    )
    cache_compressed = HttpReader(cache_compressed).end_caching(same_filepath_fn=True)
    # cache decompressed files
    cache_decompressed = cache_compressed.on_disk_cache(
        filepath_fn=Compose([partial(str.split, sep=suffix), itemgetter(0)]),
    )
    cache_decompressed = cache_decompressed.open_files(mode="b").load_from_bz2().end_caching(
        same_filepath_fn=True
    )
    return cache_decompressed

def read_bal_data(file_name: str, use_quat=False) -> dict:
    """
    Read a Bundle Adjustment in the Large dataset.

    Referenced Scipy's BAL loader: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

    According to BAL official documentation, each problem is provided as a text file in the following format:

    <num_cameras> <num_points> <num_observations>
    <camera_index_1> <point_index_1> <x_1> <y_1>
    ...
    <camera_index_num_observations> <point_index_num_observations> <x_num_observations> <y_num_observations>
    <camera_1>
    ...
    <camera_num_cameras>
    <point_1>
    ...
    <point_num_points>

    Where, there camera and point indices start from 0. Each camera is a set of 9 parameters - R,t,f,k1 and k2. The rotation R is specified as a Rodrigues' vector.

    Parameters
    ----------
    file_name : str
        The decompressed file of the dataset.

    Returns
    -------
    dict
        A dictionary containing the following fields:
        - problem_name: str
            The name of the problem.
        - camera_params: torch.Tensor (n_cameras, 9 or 10)
            contains camera parameters for each camera. If use_quat is True, the shape is (n_cameras, 10).
        - points_3d: torch.Tensor (n_points, 3)
            contains initial estimates of point coordinates in the world frame.
        - points_2d: torch.Tensor (n_observations, 2)
            contains measured 2-D coordinates of points projected on images in each observations.
        - camera_index_of_observations: torch.Tensor (n_observations,)
            contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
        - point_index_of_observations: torch.Tensor (n_observations,)
            contains indices of points (from 0 to n_points - 1) involved in each observation.
    """
    with open(file_name, "r") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = torch.empty(n_observations, dtype=torch.int64)
        point_indices = torch.empty(n_observations, dtype=torch.int64)
        points_2d = torch.empty((n_observations, 2), dtype=DTYPE)

        for i in range(n_observations):
            tmp_line = file.readline()
            camera_index, point_index, x, y = tmp_line.split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i, 0] = float(x)
            points_2d[i, 1] = float(y)

        camera_params = torch.empty(n_cameras * 9, dtype=DTYPE)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = torch.empty(n_points * 3, dtype=DTYPE)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))
    
    if use_quat:
        # convert Rodrigues vector to unit quaternion for camera rotation
        # camera_params[0:3] is the Rodrigues vector
        # after conversion, camera_params[0:4] is the unit quaternion
        # r = Rotation.from_rotvec(camera_params[:, :3])
        # q = r.as_quat()
        r = pp.so3(camera_params[:, :3])
        q = r.Exp()
        # [tx, ty, tz, q0, q1, q2, q3, f, k1, k2]
        camera_params = torch.cat([camera_params[:, 3:6], q, camera_params[:, 6:]], axis=1)
    else:
        camera_params = torch.cat([camera_params[:, 3:6], camera_params[:, :3], camera_params[:, 6:]], axis=1)

    # convert camera_params to torch.Tensor
    camera_params = torch.tensor(camera_params).to(DTYPE)

    return {'problem_name': os.path.splitext(os.path.basename(file_name))[0], # str
            'camera_params': camera_params, # torch.Tensor (n_cameras, 9 or 10)
            'points_3d': points_3d, # torch.Tensor (n_points, 3)
            'points_2d': points_2d, # torch.Tensor (n_observations, 2)
            'camera_index_of_observations': camera_indices, # torch.Tensor (n_observations,)
            'point_index_of_observations': point_indices, # torch.Tensor (n_observations,)
            }

def build_pipeline(dataset='ladybug', cache_dir='bal_data', use_quat=False):
    """
    Build a pipeline for the Bundle Adjustment in the Large dataset.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset, by default 'ladybug'.
        Must be one of ['ladybug', 'trafalgar', 'dubrovnik', 'venice', 'final'].
    cache_dir : str, optional
        The directory to cache the downloaded files, by default 'bal_data'.

    Returns
    -------
    dp : torchdata.datapipes.IterableWrapper
        The pipeline for the dataset.
        In each iteration, return a dictionary containing the following fields:
        - problem_name: str
            The name of the problem.
        - camera_params: torch.Tensor (n_cameras, 9 or 10)
            contains camera parameters for each camera. If use_quat is True, the shape is (n_cameras, 10).
        - points_3d: torch.Tensor (n_points, 3)
            contains initial estimates of point coordinates in the world frame.
        - points_2d: torch.Tensor (n_observations, 2)
            contains measured 2-D coordinates of points projected on images in each observations.
        - camera_index_of_observations: torch.Tensor (n_observations,)
            contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
        - point_index_of_observations: torch.Tensor (n_observations,)
            contains indices of points (from 0 to n_points - 1) involved in each observation.
    """
    global ALL_DATASETS
    print(f"Streaming data for {dataset}...")
    assert dataset in ALL_DATASETS, f"dataset_name must be one of {ALL_DATASETS}"
    url_dp = _problem_lister(_with_base_url(dataset + '.html'), cache_dir=cache_dir)
    download_dp = _download_pipe(cache_dir=cache_dir, url_dp=url_dp, suffix='.bz2')
    bal_data_dp = download_dp.map(partial(read_bal_data, use_quat=use_quat))
    return bal_data_dp

def get_problem(problem_name, dataset, cache_dir='bal_data', use_quat=False):
    global ALL_DATASETS
    print(f"Streaming data for {dataset}...")
    assert dataset in ALL_DATASETS, f"dataset_name must be one of {ALL_DATASETS}"
    url_dp = _problem_lister(_with_base_url(dataset + '.html'), cache_dir=cache_dir)
    def filter_problem(x):
        basename = os.path.basename(x)
        return basename in {problem_name, problem_name + '.txt', problem_name + '.txt.bz2'}
    url_dp = url_dp.filter(filter_problem)
    download_dp = _download_pipe(cache_dir=cache_dir, url_dp=url_dp, suffix='.bz2')
    bal_data_dp = download_dp.map(partial(read_bal_data, use_quat=use_quat))
    dataset_iterator = iter(bal_data_dp)
    try:
        problem = next(dataset_iterator)
    except StopIteration:
        raise ValueError(f"Problem {problem_name} not found in dataset {dataset}.")
    return problem

def _test():
    dp = build_pipeline()
    print("Testing dataset pipeline with use_quat=False...")
    for i in dp:
        point_indices = i['point_index_of_observations']
        camera_indices = i['camera_index_of_observations']
        points = i['points_3d'][point_indices]
        pixels = i['points_2d']
        camera_params = i['camera_params'][camera_indices]
        problem_name = i['problem_name']
        # check shape as in pp.reprojerr
        assert points.size(-1) == 3 and pixels.size(-1) == 2 and camera_params.size(-1) == 9, "Shape not compatible."
        # check shape at index 0, should be n_observation
        assert points.size(0) == pixels.size(0) == camera_params.size(0), "Shape not compatible."
        # check dtype is float64
        assert DTYPE == points.dtype == pixels.dtype == camera_params.dtype, "dtype not float64."
        print(problem_name, 'ok')
    
    for dataset in ALL_DATASETS:
        dp = build_pipeline(dataset=dataset, use_quat=True)
        print("Testing dataset pipeline with use_quat=True...")
        for i in dp:
            camera_params = i['camera_params']
            assert camera_params.size(-1) == 10, "Shape not compatible."
            assert DTYPE == camera_params.dtype, "dtype not float64."
            # test if the quaternion is unit
            q = camera_params[:, :4]
            # assert torch.allclose(torch.norm(q, dim=1), torch.ones(q.size(0))), "Quaternion is not unit."
            print(i['problem_name'], 'ok')
        print("All tests passed!")

if __name__ == '__main__':
    _test()
