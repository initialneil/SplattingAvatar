import os
import pickle
from pathlib import Path
import numpy as np
import cv2
import h5py
import tqdm
import torch
import json

def load_pkl(fpath):
    with open(fpath, "rb") as f:
        return pickle.load(f, encoding="latin")

def load_h5py(fpath):
    return h5py.File(fpath, "r")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="path to the PeopleSnapshotData")
    parser.add_argument("--subject", type=str, default="male-3-casual", help="sequence to process")
    parser.add_argument("--outdir", type=str, default=None, help="path to output")
    args = parser.parse_args()

    dirpath = os.path.join(args.root, args.subject)
    assert os.path.exists(dirpath), f"Cannot open {dirpath}"
    dirpath = Path(dirpath)

    if args.outdir is None:
        outdir = Path(f"/mnt/v/Dataset/PeopleSnapshot/instant_avatar/{args.subject}/")
    else:
        outdir = Path(args.outdir) / args.subject
    os.makedirs(outdir, exist_ok=True)

    # load camera
    camera = load_pkl(dirpath / "camera.pkl")
    K = np.eye(3)
    K[0, 0] = camera["camera_f"][0]
    K[1, 1] = camera["camera_f"][1]
    K[:2, 2] = camera["camera_c"]
    dist_coeffs = camera["camera_k"]

    H, W = camera["height"], camera["width"]
    w2c = np.eye(4)
    w2c[:3, :3] = cv2.Rodrigues(camera["camera_rt"])[0]
    w2c[:3, 3] = camera["camera_t"]

    camera_path = outdir / "cameras.npz"
    np.savez(str(camera_path), **{
        "intrinsic": K,
        "extrinsic": w2c,
        "height": H,
        "width": W,
    })
    print("Write camera to", camera_path)
    with open(str(camera_path).replace('.npz', '.json'), 'w') as fp:
        json.dump({
            "intrinsic": K.tolist(),
            "extrinsic": w2c.tolist(),
            "height": H,
            "width": W,
        }, fp)
    torch.save({
        "intrinsic": K,
        "extrinsic": w2c,
        "height": H,
        "width": W,
    }, str(camera_path).replace('.npz', '.pt'))

    # load images
    image_dir = outdir / "images"
    os.makedirs(image_dir, exist_ok=True)

    # print("Write images to", image_dir)
    # cap = cv2.VideoCapture(str(dirpath / f"{args.subject}.mp4"))
    # frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # for i in tqdm.trange(frame_cnt):
    #     img_path = f"{image_dir}/image_{i:04d}.png"
    #     ok, frame = cap.read()
    #     if not ok: break
    #     frame = cv2.undistort(frame, K, dist_coeffs)
    #     cv2.imwrite(img_path, frame)

    # load masks
    mask_dir = outdir / "masks"
    os.makedirs(mask_dir, exist_ok=True)

    print("Write mask to", mask_dir)
    masks = np.asarray(load_h5py(dirpath / "masks.hdf5")["masks"]).astype(np.uint8)
    for i, mask in enumerate(tqdm.tqdm(masks)):
        mask_path = f"{mask_dir}/mask_{i:04d}.npy"
        mask = cv2.undistort(mask, K, dist_coeffs)
        # np.save(mask_path, mask)
        # cv2.imwrite(mask_path.replace('.npy', '.png'), mask * 255)
 
        # remove boundary artifact
        alpha = mask * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        alpha = cv2.erode(alpha, kernel)
        alpha = cv2.blur(alpha, (3, 3))
        cv2.imwrite(mask_path.replace('.npy', '.png'), alpha)

    smpl_params = load_h5py(dirpath / "reconstructed_poses.hdf5")
    smpl_params = {
        "betas": np.asarray(smpl_params["betas"]).astype(np.float32),
        "thetas": np.asarray(smpl_params["pose"]).astype(np.float32),
        "transl": np.asarray(smpl_params["trans"]).astype(np.float32),
    }
    np.savez(str(outdir / "poses.npz"), **smpl_params)

    smpl_params = {
        "betas": torch.tensor(smpl_params["betas"]),
        "thetas": torch.tensor(smpl_params["thetas"]),
        "transl": torch.tensor(smpl_params["transl"]),
    }
    torch.save(smpl_params, str(outdir / "poses.pt"))
