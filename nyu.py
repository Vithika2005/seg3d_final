"""NYU Depth V2 labeled subset loader (1449 RGB-D images with class + instance labels)."""
import os
import h5py
import numpy as np

CROP = (45, 471, 41, 601)  # standard valid region (rows, cols)
FX, FY, CX, CY = 518.8579, 519.4696, 325.5824, 253.7362
NYU_VOCAB = ["chair", "table", "sofa", "bed", "cabinet", "bookshelf", "desk", "door",
             "window", "pillow", "lamp", "television"]


class NYULabeled:
    def __init__(self, path):
        if not os.path.exists(path) or not h5py.is_hdf5(path):
            gb = os.path.getsize(path) / 1e9 if os.path.exists(path) else 0
            raise RuntimeError(f"{path} is not a valid NYU labeled .mat (HDF5) file (found {gb:.2f} GB; the real file is "
                               "about 2.8 GB). The download most likely failed or saved a web page - see README section 3.")
        self.f = h5py.File(path, "r")
        self.n = self.f["images"].shape[0]
        self.names = ["".join(chr(c) for c in self.f[r][:].flatten()) for r in self.f["names"][0]]

    def get(self, i):
        r0, r1, c0, c1 = CROP
        t = lambda key: np.asarray(self.f[key][i]).T[r0:r1, c0:c1]
        rgb = np.transpose(np.asarray(self.f["images"][i]), (2, 1, 0))[r0:r1, c0:c1]
        K = np.array([[FX, 0, CX - c0], [0, FY, CY - r0], [0, 0, 1]])
        return {"rgb": np.ascontiguousarray(rgb), "depth": t("depths").astype(np.float32),
                "labels": t("labels").astype(np.int64), "instances": t("instances").astype(np.int64), "K": K}
