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

def qr_decomposition(A):
    if isinstance(A, np.ndarray):
        A = A.tolist()
    
    n = len(A)
    m = len(A[0])
    
    A_cols = []
    for j in range(m):
        col = [float(A[i][j]) for i in range(n)] 
        A_cols.append(col)
    
    Q_cols = []
    R = [[0.0 for _ in range(m)] for _ in range(m)]
    
    for j in range(m):
        v = A_cols[j][:]
        for i in range(j):
            r_ij = sum(Q_cols[i][k] * A_cols[j][k] for k in range(n))
            R[i][j] = r_ij
            v = [v[k] - r_ij * Q_cols[i][k] for k in range(n)]
        
        norm_v = math.sqrt(sum(float(x)**2 for x in v))
        R[j][j] = norm_v
        
        q = [float(x) / norm_v for x in v] if norm_v > 1e-12 else [0.0]*n
        Q_cols.append(q)
    
    Q = [[Q_cols[j][i] for j in range(m)] for i in range(n)]
    
    return Q, R

def rq_decomposition(A):
    if isinstance(A, np.ndarray):
        A = A.tolist()
    
    A_rev = [row[::-1] for row in A[::-1]]
    Q_rev, R_rev = qr_decomposition(A_rev)
    R = [row[::-1] for row in R_rev[::-1]]
    Q = [row[::-1] for row in Q_rev[::-1]]
    return R, Q

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
    # Compute centroid.
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n

    # Shift points so that centroid is at the origin.
    shifted = [[p[0] - cx, p[1] - cy] for p in points]
    
    # Compute the average Euclidean distance to the origin.
    avg_dist = sum(math.sqrt(p[0]**2 + p[1]**2) for p in shifted) / n
    scale = target / avg_dist

    # Scale the points.
    normalized = [[scale * p[0], scale * p[1]] for p in shifted]
    
    # Build the normalization matrix T such that for a homogeneous point [x, y, 1]:
    # [x_norm, y_norm, 1]^T = T * [x, y, 1]^T
    T = [
        [scale,    0, -scale * cx],
        [   0, scale, -scale * cy],
        [   0,    0,           1]
    ]
    return normalized, T

# === End Custom Routines ===
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
        return vector  # or return a default vector; here we simply return it.
    return [x / vector_norm for x in vector]

def eigen_decomposition(matrix, tol=1e-15, max_iterations=100000):
    """
    Compute eigenvalues and eigenvectors using the QR algorithm.
    Returns (eigenvalues, eigenvectors) where eigenvectors are the columns of V.

    DOES NOT WORK RIGHT NOW. It drifts to zero.
    """
    n = len(matrix)
    # Make a deep copy of matrix.
    A = [row[:] for row in matrix]
    V = identity_matrix(n)
    for _ in range(max_iterations):
        Q, R = qr_decomposition(A)
        A = mat_mult(R, Q)
        V = mat_mult(V, Q)
        off_diagonal = sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j) ** 0.5
        if off_diagonal < tol:
            break
    # The eigenvalues are on the diagonal of A.
    eigenvalues = [A[i][i] for i in range(n)]
    # The eigenvectors are given by the columns of V.
    # We use safe_normalize on each column.
    eigenvectors = []
    for i in range(n):
        col = [V[j][i] for j in range(n)]
        eigenvectors.append(safe_normalize(col))
    return eigenvalues, eigenvectors


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
    # Use NumPy SVD on the 2N x 9 matrix A.
    # U, S, VT = np.linalg.svd(A)
    U, S, VT = svd_custom(A)
    S = np.array(S, dtype=np.float64)
    S = np.diag(S)
    VT = np.array(VT, dtype=np.float64)
    h = VT[-1, :]
    H = h.reshape(3,3)
    # Normalize so that H[2,2] = 1
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
    # Prepare object points (the planar grid) in 2D.
    objp = np.zeros((board_size[0]*board_size[1], 2), np.float64)
    # Use np.mgrid to create grid of (x,y) coordinates.
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # scale to real-world units
    # Normalize object points
    norm_obj, T_obj = normalize_points_2d(objp.tolist(), target=math.sqrt(2))
    
    homographies = []  # will store the homography H for each image
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
        # Refine corners for subpixel accuracy.
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        # corners comes as an array of shape (N,1,2). Convert to list of [u,v].
        img_pts = [corner[0].tolist() for corner in corners]
        # Normalize image points.
        norm_img, T_img = normalize_points_2d(img_pts, target=math.sqrt(2))
        # Compute homography H that maps normalized object points to normalized image points.
        H = compute_homography(norm_obj, norm_img)
        # Denormalize the homography:
        # The relation is: H_original = T_img^{-1} * H * T_obj
        T_img_inv = np.linalg.inv(np.array(T_img, dtype=np.float64))
        H_orig = T_img_inv.dot(H).dot(np.array(T_obj, dtype=np.float64))
        # Normalize H so that H[2,2] = 1.
        H_orig = H_orig / H_orig[2,2]
        homographies.append(H_orig)
        # Optionally, show detected corners.
        cv2.drawChessboardCorners(img, board_size, corners, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    if len(homographies) < 1:
        print("No valid homographies computed!")
        return None, None
    
    # ---------------------------
    # Build the constraint matrix V for intrinsic parameters.
    # For each homography H, let h1, h2 be its first two columns.
    # For each image, add two equations:
    #    v_12 = 0
    #    v_11 - v_22 = 0
    # where v_ij are defined as in Zhang's method.
    # ---------------------------
    V = []
    for H in homographies:
        # Extract columns from H:
        h1 = H[:,0]
        h2 = H[:,1]
        # Compute v12, v11, v22 using the definitions:
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
    # Solve V * b = 0 using SVD.
    U, S, VT = svd_custom(V)
    S = np.array(S, dtype=np.float64)
    S = np.diag(S)
    VT = np.array(VT, dtype=np.float64)
    b = VT[-1, :]  # solution corresponding to smallest singular value.
    # b corresponds to the symmetric matrix B = [B11, B12, B22, B13, B23, B33]
    B11, B12, B22, B13, B23, B33 = b
    # Recover intrinsic parameters from B using the formulas from Zhang's paper.
    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lambda_val = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    if lambda_val < 0:
        print("Negative lambda encountered. Calibration failed.")
        lambda_val = 0.000000001
    alpha = math.sqrt(lambda_val / B11)
    if (B22 - B12**2 / B11) < 0:
        print("Negative value encountered in beta calculation. Calibration failed.")
        beta = alpha
    else:
        beta  = math.sqrt(lambda_val * B11 / (B11*B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lambda_val
    u0 = gamma * v0 / beta - B13 * alpha**2 / lambda_val
    # Build the intrinsic matrix K.
    K = np.array([
        [alpha, gamma, u0],
        [0,     beta,  v0],
        [0,     0,     1]
    ], dtype=np.float64)
    print("Estimated Intrinsic Matrix K:")
    print(K)
    
    # ---------------------------
    # Compute extrinsics for each image.
    # For each homography H, we have: H = K [r1 r2 t]  (up to scale).
    # We compute:
    #    lambda = 1/||K^{-1}h1||
    #    r1 = lambda * K^{-1}h1
    #    r2 = lambda * K^{-1}h2
    #    r3 = r1 x r2
    #    t  = lambda * K^{-1}h3
    # ---------------------------
    extrinsics = []
    K_inv = np.linalg.inv(K)
    for H in homographies:
        h1 = H[:,0]
        h2 = H[:,1]
        h3 = H[:,2]
        # Compute scale lambda.
        lam = 1.0 / np.linalg.norm(K_inv.dot(h1))
        r1 = lam * K_inv.dot(h1)
        r2 = lam * K_inv.dot(h2)
        # Ensure orthogonality by computing r3 = r1 x r2.
        r3 = np.cross(r1, r2)
        t = lam * K_inv.dot(h3)
        R = np.column_stack((r1, r2, r3))
        extrinsics.append((R, t))
    return K, extrinsics

def svd_custom(A):
    """
    Compute the Singular Value Decomposition of matrix A.
    Returns U, singular values (as a 1D array), and V^T such that:
      A = U * diag(s) * V^T
    This routine uses the relation A^T A = V * diag(s^2) * V^T and computes U as:
      U = A * V * diag(1/s)
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    # Compute A^T A (symmetric n x n matrix)
    AtA = np.dot(A.T, A)
    # eigenvals, V = eigen_decomposition(AtA)
    eigenvals, V = np.linalg.eig(AtA)
    V = np.array(V, dtype=float)
    eigenvals = np.array(eigenvals, dtype=float)
    # Sort eigenvalues and corresponding eigenvectors
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    V = V[:, idx]
    # Singular values are the square roots of eigenvalues (clipping at 0 for safety)
    singular_vals = np.sqrt(np.maximum(eigenvals, 0))
    
    # Compute U = A * V * diag(1/s)
    U = np.dot(A, V)
    for i in range(len(singular_vals)):
        if singular_vals[i] > 1e-12:
            U[:, i] /= singular_vals[i]
        else:
            U[:, i] = 0.0
    return U, singular_vals, transpose(V)

def main():
    # -- Load calibration checkboard images -- #
    calib_dir = "calibration_images"  # Directory with calibration images
    board_size = (11, 7)  # Number of inner corners in the chessboard pattern
    square_size = 0.030  # Size of a square in meters (e.g., 2.5 cm)
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
