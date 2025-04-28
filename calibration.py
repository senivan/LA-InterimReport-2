#!/usr/bin/env python3
"""
Stereo processing with camera calibration and rectification.
This program loads a stereo pair from the Middlebury scenes2005 dataset,
uses calibration data (intrinsics and extrinsics) from your report to rectify the images,
and then computes a disparity map.
Custom linear algebra routines (e.g., RQ decomposition) are used in the calibration step.
OpenCV is used for image processing.
"""

import cv2
import numpy as np
import math
import os


def mat_mult(A, B):
    n = len(A)
    p = len(A[0])
    m = len(B[0])
    C = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def transpose(A):
    n = len(A)
    m = len(A[0])
    T = [[A[i][j] for i in range(n)] for j in range(m)]
    return T

def cross_product(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]

def normalize_points_2d(points, target=15):
    """
    Normalizes a list of 2D points so that the centroid is at the origin and 
    the average distance from the origin equals target.
    Returns (normalized_points, T) where T is the 3x3 normalization matrix.
    """
    n = len(points)

    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n

    shifted = [[p[0] - cx, p[1] - cy] for p in points]
    
    avg_dist = sum(math.sqrt(p[0]**2 + p[1]**2) for p in shifted) / n
    scale = target / avg_dist

    normalized = [[scale * p[0], scale * p[1]] for p in shifted]
    
    T = [
        [scale,    0, -scale * cx],
        [   0, scale, -scale * cy],
        [   0,    0,           1]
    ]
    return normalized, T

def identity_matrix(size):
    """Create an identity matrix of given size."""
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]

def norm(vector):
    """Calculate the Euclidean norm of a vector."""
    return sum(x ** 2 for x in vector) ** 0.5

def safe_normalize(vector, tol=1e-12):
    """Normalize vector safely; if nearly zero, return the vector as is."""
    vector_norm = norm(vector)
    if vector_norm < tol:
        return vector
    return [x / vector_norm for x in vector]

def compute_homography(obj_pts, img_pts):
    """
    Compute a 3x3 homography that maps 2D object points (on the calibration plane)
    to 2D image points. obj_pts and img_pts should be lists of [x, y] (already normalized if desired).
    Uses the Direct Linear Transform (DLT) method.
    We build a 2N x 9 matrix A and solve A*h=0 using SVD.
    Returns H as a 3x3 NumPy array (normalized so that H[2,2]=1).
    """
    N = len(obj_pts)
    A = []
    for i in range(N):
        X, Y = obj_pts[i]
        u, v = img_pts[i]
        A.append([-X, -Y, -1, 0, 0, 0, u*X, u*Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
    A = np.array(A, dtype=np.float64)

    U, S, VT = svd_custom(A)
    S = np.array(S, dtype=np.float64)
    S = np.diag(S)
    VT = np.array(VT, dtype=np.float64)
    h = VT[-1, :]
    H = h.reshape(3,3)

    if abs(H[2,2]) > 1e-12:
        H = H / H[2,2]
    return H


# ---------------------------
# Multi-Image Calibration Function
# ---------------------------

def multi_image_calibration(calib_dir, board_size, square_size):
    """
    Perform camera calibration from multiple images of a planar chessboard.
    
    Parameters:
      calib_dir: Directory containing calibration images.
      board_size: Tuple (number of inner corners along width, height).
      square_size: Real-world size of one square (e.g. in meters).
    
    Returns:
      K: The estimated 3x3 intrinsic matrix.
      extrinsics: A list of (R, t) for each image.
    """
    objp = np.zeros((board_size[0]*board_size[1], 2), np.float64)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp = objp * square_size
    norm_obj, T_obj = normalize_points_2d(objp.tolist(), target=math.sqrt(2))
    
    homographies = []
    image_files = [f for f in os.listdir(calib_dir) if f.lower().endswith((".png",".jpg"))]
    if not image_files:
        print("No calibration images found in directory.")
        return None, None
    
    for fname in image_files:
        img = cv2.imread(os.path.join(calib_dir, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not ret:
            print(f"Chessboard not found in {fname}. Skipping.")
            continue

        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        img_pts = [corner[0].tolist() for corner in corners]
        norm_img, T_img = normalize_points_2d(img_pts, target=math.sqrt(2))
        H = compute_homography(norm_obj, norm_img)

        T_img_inv = np.linalg.inv(np.array(T_img, dtype=np.float64))
        H_orig = T_img_inv.dot(H).dot(np.array(T_obj, dtype=np.float64))

        H_orig = H_orig / H_orig[2,2]
        homographies.append(H_orig)

        cv2.drawChessboardCorners(img, board_size, corners, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    if len(homographies) < 1:
        print("No valid homographies computed!")
        return None, None
    
    V = []
    for H in homographies:
        h1 = H[:,0]
        h2 = H[:,1]
        v12 = np.array([h1[0]*h2[0],
                        h1[0]*h2[1] + h1[1]*h2[0],
                        h1[1]*h2[1],
                        h1[2]*h2[0] + h1[0]*h2[2],
                        h1[2]*h2[1] + h1[1]*h2[2],
                        h1[2]*h2[2]], dtype=np.float64)
        v11 = np.array([h1[0]*h1[0],
                        h1[0]*h1[1] + h1[1]*h1[0],
                        h1[1]*h1[1],
                        h1[2]*h1[0] + h1[0]*h1[2],
                        h1[2]*h1[1] + h1[1]*h1[2],
                        h1[2]*h1[2]], dtype=np.float64)
        v22 = np.array([h2[0]*h2[0],
                        h2[0]*h2[1] + h2[1]*h2[0],
                        h2[1]*h2[1],
                        h2[2]*h2[0] + h2[0]*h2[2],
                        h2[2]*h2[1] + h2[1]*h2[2],
                        h2[2]*h2[2]], dtype=np.float64)
        V.append(v12)
        V.append(v11 - v22)
    V = np.array(V, dtype=np.float64)

    U, S, VT = svd_custom(V)
    S = np.array(S, dtype=np.float64)
    S = np.diag(S)
    VT = np.array(VT, dtype=np.float64)
    b = VT[-1, :]
    B11, B12, B22, B13, B23, B33 = b
    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lambda_val = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    if lambda_val < 0:
        print("Negative lambda encountered. Calibration failed.")
        lambda_val = 0.000000001
    if abs(B11) < 1e-12:
        print("B11 is nearly zero. Calibration may be invalid. Setting B11 to a small value.")
        B11 = 1e-9
    if lambda_val / B11 < 0:
        print("Invalid value for lambda_val / B11. Calibration failed.")
        alpha = 0
    else:
        alpha = math.sqrt(lambda_val / B11)
    if (B22 - B12**2 / B11) < 0:
        print("Negative value encountered in beta calculation. Calibration failed.")
        beta = alpha
    else:
        beta  = math.sqrt(lambda_val * B11 / (B11*B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lambda_val
    u0 = gamma * v0 / beta - B13 * alpha**2 / lambda_val

    K = np.array([
        [alpha, gamma, u0],
        [0,     beta,  v0],
        [0,     0,     1]
    ], dtype=np.float64)
    print("Estimated Intrinsic Matrix K:")
    print(K)
    
    extrinsics = []
    K_inv = np.linalg.inv(K)
    for H in homographies:
        h1 = H[:,0]
        h2 = H[:,1]
        h3 = H[:,2]
        lam = 1.0 / np.linalg.norm(K_inv.dot(h1))
        r1 = lam * K_inv.dot(h1)
        r2 = lam * K_inv.dot(h2)
        r3 = np.cross(r1, r2)
        t = lam * K_inv.dot(h3)
        R = np.column_stack((r1, r2, r3))
        extrinsics.append((R, t))
    return K, extrinsics

def power_iteration(A, num_iter=1000, eps=1e-12):
    v = np.random.randn(A.shape[1])
    for _ in range(num_iter):
        v = A @ v
        v /= np.linalg.norm(v) + eps
    return v / np.linalg.norm(v)

def svd_custom(A, num_iter=1000, eps=1e-12):
    """
    Compute the Singular Value Decomposition of matrix A using power iteration 
    and deflation to compute the eigenvalues and eigenvectors of A^T A.

    Returns:
      U, singular values (as a 1D array), and V^T such that:
          A = U * diag(s) * V^T
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    AtA = np.dot(A.T, A)
    
    M = AtA.copy()
    eigenvals = []
    eigenvecs = []
    for _ in range(n):
        v = power_iteration(M, num_iter=num_iter, eps=eps)
        eigen_val = np.dot(v, np.dot(M, v))
        eigenvals.append(eigen_val)
        eigenvecs.append(v)
        M = M - eigen_val * np.outer(v, v)
    
    eigenvals = np.array(eigenvals, dtype=float)
    V = np.array(eigenvecs, dtype=float).T

    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    V = V[:, idx]

    singular_vals = np.sqrt(np.maximum(eigenvals, 0))
    
    U = np.dot(A, V)
    for i in range(len(singular_vals)):
        if singular_vals[i] > eps:
            U[:, i] /= singular_vals[i]
        else:
            U[:, i] = 0.0
    return U, singular_vals, transpose(V)

def main():
    calib_dir = "prod_calib"
    board_size = (8, 6)
    square_size = 0.030
    K, extrinsics = multi_image_calibration(calib_dir, board_size, square_size)
    if K is None or extrinsics is None:
        print("Calibration failed.")
    
    print("Extrinsics")
    for i, (R, t) in enumerate(extrinsics):
        print(f"Image {i}:")
        print("Rotation Matrix R:")
        print(R)
        print("Translation Vector t:")
        print(t)

    

if __name__ == "__main__":
    main()
