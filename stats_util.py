from math import sqrt


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(v):
    """v = [n, top1_count, anyhit_count, returned_total]"""
    n, t1, ah, ret = v
    lo, hi = wilson(t1, n)
    return {"n": n, "top1": t1 / n if n else 0.0, "any_hit": ah / n if n else 0.0,
            "avg_returned": ret / n if n else 0.0, "top1_ci95": [round(lo, 3), round(hi, 3)]}
