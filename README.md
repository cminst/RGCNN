## Regularized Graph CNN for Point Cloud Segmentation

This code is a PyTorch implementation of [RGCNN: Regularized Graph CNN for Point Cloud Segmentation][arxiv], ACM MultiMedia, 2018.

## Installation

This code runs on PyTorch with Python 3 and standard scientific Python dependencies (numpy, scipy, scikit-learn). We borrow the framework of [cnn_graph][cng].

## Usage

It requires original ModelNet40 and ShapeNet data, which can be downloaded [here][data_seg] for segmentation and [here][data_cls] for classification. You can use the tool provided by [pointnet][pointnet++] to convert the data to numpy array. We also provide our [processed one][data_pre] but we don't guarantee its compatibility.

The `train.py` script trains either the segmentation or classification model from numpy arrays:

```
python train.py --task seg --data-dir /path/to/data
```

For classification, pass `--task cls` and ensure `label_{split}.npy` contains class indices.

## Note

We test our code on jupyter notebook at first, so I suspect some part could be missing. If something went wrong, please contact me by email.

## Preparing ModelNet40 numpy files

If you have the official `modelnet40_normal_resampled` folder (class subdirs plus `modelnet40_train.txt`/`modelnet40_test.txt`), convert it to the `.npy` format expected by `train.py` with:

```
python3 convert_modelnet40_rgcnn.py \
  --root modelnet40_normal_resampled \
  --out-dir data \
  --num-points 1024 \
  --val-ratio 0.1
```

This produces `data_train.npy`, `label_train.npy`, `data_val.npy`, `label_val.npy`, and `data_test.npy` under `./data`.

[cng]: https://github.com/mdeff/cnn_graph
[arxiv]: https://arxiv.org/abs/1806.02952
[data_seg]: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
[data_cls]: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
[pointnet++]: https://github.com/charlesq34/pointnet2
[data_pre]: https://1drv.ms/f/s!Am_uh1epJzCIjQeZviRjHa4fCkFy
