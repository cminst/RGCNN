import argparse
import pathlib
import glob

import h5py
import numpy as np


def resample_points(pts, num_points, rng):
    """Downsample (or upsample with replacement) a single point cloud."""
    n = pts.shape[0]
    if n == num_points:
        return pts
    if n > num_points:
        idx = rng.choice(n, num_points, replace=False)
    else:
        idx = rng.choice(n, num_points, replace=True)
    return pts[idx]


def load_h5_split(files, num_points, rng):
    datas, labels = [], []
    for f in files:
        with h5py.File(f, "r") as h5:
            data = h5["data"][:]  # [N, P, 6]
            lab = h5["label"][:].reshape(-1).astype(np.int64)
            data = resample_points(data, num_points, rng)
            datas.append(data)
            labels.append(lab)
    return np.concatenate(datas, axis=0), np.concatenate(labels, axis=0)


def load_txt_split(root, split_name, num_points, rng):
    """Load ModelNet40 normals from the raw txt tree (class folders)."""
    shape_names = (pathlib.Path(root) / "modelnet40_shape_names.txt").read_text().strip().splitlines()
    class_to_idx = {name: i for i, name in enumerate(shape_names)}

    split_file = pathlib.Path(root) / f"modelnet40_{split_name}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")

    files = [line.strip() for line in split_file.read_text().strip().splitlines() if line.strip()]
    datas, labels = [], []
    for rel in files:
        cls = rel.split("/")[0]
        label = class_to_idx[cls]
        txt_path = pathlib.Path(root) / f"{rel}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing point cloud file: {txt_path}")
        pts = np.loadtxt(txt_path).astype(np.float32)  # [P, 6] (xyz + normals)
        if pts.shape[1] < 6:
            raise ValueError(f"Expected 6 columns (xyz+normals) in {txt_path}, got {pts.shape[1]}")
        pts = resample_points(pts, num_points, rng)
        datas.append(pts)
        labels.append(label)
    return np.stack(datas, axis=0), np.array(labels, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description="Convert ModelNet40 normals into RGCNN numpy format.")
    ap.add_argument("--root", default="data/modelnet40_normal_resampled",
                    help="Directory containing train*.h5/test*.h5 OR the raw txt folders + modelnet40_*.txt files")
    ap.add_argument("--out-dir", default="data", help="Output directory for *.npy files")
    ap.add_argument("--num-points", type=int, default=2048, help="Target point count (e.g., 1024 for the paper)")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of training set for validation split")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for downsampling and train/val split")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    rng = np.random.RandomState(args.seed)

    train_files = sorted(glob.glob(str(root / "train*.h5")))
    test_files = sorted(glob.glob(str(root / "test*.h5")))

    if train_files and test_files:
        print("Detected H5 format; loading train*.h5 / test*.h5 ...")
        train_data, train_labels = load_h5_split(train_files, args.num_points, rng)
        test_data, test_labels = load_h5_split(test_files, args.num_points, rng)
    else:
        print("Detected raw txt format; loading from class folders...")
        train_data, train_labels = load_txt_split(root, "train", args.num_points, rng)
        test_data, test_labels = load_txt_split(root, "test", args.num_points, rng)

    n = train_data.shape[0]
    n_val = int(round(n * args.val_ratio))
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "data_train.npy", train_data[tr_idx])
    np.save(out_dir / "label_train.npy", train_labels[tr_idx])
    np.save(out_dir / "data_val.npy", train_data[val_idx])
    np.save(out_dir / "label_val.npy", train_labels[val_idx])

    np.save(out_dir / "data_test.npy", test_data)
    np.save(out_dir / "label_test.npy", test_labels)

    print(f"Saved train/val/test splits to {out_dir.resolve()}")
    print("Shapes:")
    print("  train:", train_data[tr_idx].shape, train_labels[tr_idx].shape)
    print("  val  :", train_data[val_idx].shape, train_labels[val_idx].shape)
    print("  test :", test_data.shape, test_labels.shape)


if __name__ == "__main__":
    main()
