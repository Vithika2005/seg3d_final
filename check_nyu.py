"""Full integrity check of nyu_depth_v2_labeled.mat. Reads every image / depth / label / instance slice
(the data is compressed, so damaged chunks raise errors). Usage: python check_nyu.py [path]"""
import os, sys, time
import h5py

EXPECTED_SIZE, EXPECTED_N = 2972037809, 1449


def check(path, expected_size=EXPECTED_SIZE, n_expected=EXPECTED_N, verbose=True):
    size = os.path.getsize(path)
    if verbose:
        print(f"size: {size} bytes (expected {expected_size})")
    if expected_size and size != expected_size:
        return False, f"wrong size ({size}); re-run download_nyu.sh"
    try:
        f = h5py.File(path, "r")
        for k in ("images", "depths", "labels", "instances", "names"):
            if k not in f:
                return False, f"missing dataset '{k}'"
        n = f["images"].shape[0]
        if n_expected and n != n_expected:
            return False, f"expected {n_expected} images, found {n}"
        bad = []; t0 = time.time()
        for k in ("images", "depths", "labels", "instances"):
            for i in range(n):
                try:
                    f[k][i]
                except Exception as e:
                    bad.append((k, i, str(e)[:70]))
                if verbose and i % 100 == 0:
                    print(f"\r{k}: {i}/{n}  ({time.time() - t0:.0f}s)", end="", flush=True)
            if verbose:
                print(f"\r{k}: {n}/{n} checked" + " " * 20)
        if bad:
            return False, f"{len(bad)} unreadable slices, first: {bad[:3]}"
    except Exception as e:
        return False, f"cannot read file: {e}"
    return True, f"{n} images readable"


if __name__ == "__main__":
    ok, msg = check(sys.argv[1] if len(sys.argv) > 1 else "data/nyu_depth_v2_labeled.mat")
    print(("PASS: " if ok else "FAIL: ") + msg)
    if not ok:
        print("Delete the file and download again with:  bash download_nyu.sh \"$HOME/nyu_data\" --fresh   (or copy a good copy from a classmate)")
    sys.exit(0 if ok else 1)
