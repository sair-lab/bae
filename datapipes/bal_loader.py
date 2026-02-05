"""
This file contains the pipeline for the Bundle Adjustment in the Large dataset.

The dataset is from the following paper:
Sameer Agarwal, Noah Snavely, Steven M. Seitz, and Richard Szeliski.
Bundle adjustment in the large.
In European Conference on Computer Vision (ECCV), 2010.

Link to the dataset: https://grail.cs.washington.edu/projects/bal/
"""

import os
import warnings

import torch
from functools import partial
from operator import methodcaller

from .bal_io import DTYPE, read_bal_data

def _torchdata():
    try:
        from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper
    except ImportError as e:
        raise ImportError(
            "torchdata is required for datapipes.bal_loader streaming utilities. "
            "If you only need parsing, import read_bal_data from datapipes.bal_io."
        ) from e
    return HttpReader, IterableWrapper, FileOpener

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
    HttpReader, IterableWrapper, FileOpener = _torchdata()
    try:
        from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
    except ImportError as e:
        raise ImportError(
            "bs4 is required for datapipes.bal_loader streaming utilities. "
            "If you only need parsing, import read_bal_data from datapipes.bal_io."
        ) from e

    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

    def _cache_path(url: str) -> str:
        return os.path.join(cache_dir, os.path.basename(url))

    problem_list_dp = IterableWrapper(problem_url).on_disk_cache(
        filepath_fn=_cache_path,
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
    HttpReader, _, _ = _torchdata()

    def _cache_path(url: str) -> str:
        return os.path.join(cache_dir, os.path.basename(url))

    def _strip_suffix(path: str) -> str:
        return path.split(suffix)[0]

    # cache compressed files
    cache_compressed = url_dp.on_disk_cache(
        filepath_fn=_cache_path,
    )
    cache_compressed = HttpReader(cache_compressed).end_caching(same_filepath_fn=True)
    # cache decompressed files
    cache_decompressed = cache_compressed.on_disk_cache(
        filepath_fn=_strip_suffix,
    )
    cache_decompressed = cache_decompressed.open_files(mode="b").load_from_bz2().end_caching(
        same_filepath_fn=True
    )
    return cache_decompressed

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
