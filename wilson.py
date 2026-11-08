"""95% confidence interval for an accuracy. Usage: python wilson.py <correct> <total>   e.g. python wilson.py 26 40"""
import sys
from math import sqrt

k, n = int(sys.argv[1]), int(sys.argv[2]); z = 1.96
p = k / n; d = 1 + z * z / n
c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
print(f"accuracy {p:.2f}  (95% CI {c - h:.2f} - {c + h:.2f}, n={n})")
