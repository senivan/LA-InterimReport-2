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
def solve_homogeneous_system(A):
    m = len(A)
    n = len(A[0])
    
    M = [row[:] for row in A]
    
    row = 0
    for col in range(n):
        pivot_row = None
        max_val = 0.0
        for r in range(row, m):
            if abs(M[r][col]) > max_val:
                max_val = abs(M[r][col])
                pivot_row = r
        if pivot_row is None or abs(M[pivot_row][col]) < 1e-12:
            continue  
        M[row], M[pivot_row] = M[pivot_row], M[row]
        pivot = M[row][col]
        M[row] = [val / pivot for val in M[row]]
        for r in range(m):
            if r != row:
                factor = M[r][col]
                M[r] = [M[r][c] - factor * M[row][c] for c in range(n)]
        row += 1
        if row == m:
            break
    x = [0.0] * n
    x[n-1] = 1.0
    for r in range(m):
        pivot_col = None
        for c in range(n):
            if abs(M[r][c]) > 1e-12:
                pivot_col = c
                break
        if pivot_col is None or pivot_col == n-1:
            continue
        s = 0.0
        for j in range(pivot_col+1, n):
            s += M[r][j] * x[j]
        x[pivot_col] = -s
    return x
def solve_homogeneous_svd(A):
    """
    A is given as a list of lists.
    We convert it into a NumPy array just for the SVD step,
    then return the singular vector corresponding to the smallest singular value.
    """
    A_np = np.array(A, dtype=np.float64)
    # Compute SVD: A = U * S * V^T
    U, S, VT = np.linalg.svd(A_np)
    # The solution is the last row of VT (or last column of V)
    p = VT[-1, :]
    return p.tolist()
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


def normalize_points_3d(points, target=15):
    """
    Normalizes a list of 3D points so that the centroid is at the origin and 
    the average distance from the origin equals target.
    Returns (normalized_points, T) where T is the 4x4 normalization matrix.
    """
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    cz = sum(p[2] for p in points) / n

    shifted = [[p[0] - cx, p[1] - cy, p[2] - cz] for p in points]
    avg_dist = sum(math.sqrt(p[0]**2 + p[1]**2 + p[2]**2) for p in shifted) / n
    scale = target / avg_dist
    normalized = [[scale * p[0], scale * p[1], scale * p[2]] for p in shifted]
    
    T = [
        [scale,     0,     0, -scale * cx],
        [    0, scale,     0, -scale * cy],
        [    0,     0, scale, -scale * cz],
        [    0,     0,     0,           1]
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

def eigen_decomposition(matrix, tol=1e-10, max_iterations=1000):
    """
    Compute eigenvalues and eigenvectors using the QR algorithm.
    Returns (eigenvalues, eigenvectors) where eigenvectors are the columns of V.
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

def svd(matrix):
    """
    Compute the Singular Value Decomposition of a matrix using custom routines.
    Returns U, singular_values (as a list), and V_T such that:
        A = U * S * V^T
    where S is constructed from the singular values.
    """
    m = len(matrix)
    n = len(matrix[0])
    
    # Step 1: Compute A^T * A (an n x n matrix).
    A_T = transpose(matrix)   # n x m
    A_TA = mat_mult(A_T, matrix)  # n x n

    # Step 2: Compute eigen-decomposition of A_TA to obtain V and eigenvalues.
    eigenvalues, V = eigen_decomposition(A_TA)
    
    # Step 3: Sort eigenvalues (and corresponding eigenvectors) in descending order.
    sorted_indices = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i], reverse=True)
    eigenvalues = [eigenvalues[i] for i in sorted_indices]
    V = [V[i] for i in sorted_indices]  # Each v in V is a vector of length n.
    
    # Compute singular values (square roots of eigenvalues).
    singular_values = [math.sqrt(max(ev, 0)) for ev in eigenvalues]

    # Step 4: Compute U from A and V.
    # For each singular value (for i in range(n)), compute:
    #    u_i = A * v_i / sigma_i   (if sigma_i != 0)
    # U will have as many columns as n. If m > n, we need to complete U to an m x m orthonormal basis.
    U = []
    for i in range(n):
        v_i = V[i]
        # Compute A*v_i (result is a vector of length m).
        Av = [sum(matrix[row][k] * v_i[k] for k in range(n)) for row in range(m)]
        sigma = singular_values[i]
        if sigma > 1e-12:
            u_i = [x / sigma for x in Av]
        else:
            # For sigma = 0, we cannot determine u_i uniquely.
            # Here we simply set u_i to a zero vector.
            u_i = [0.0] * m
        U.append(u_i)
    # At this point, U is m x n. To form a full orthonormal basis, if m > n,
    # we complete U with additional orthonormal vectors.
    if m > n:
        # Using a simple Gram-Schmidt on the standard basis to fill up U.
        # First, transpose U to have columns as computed u_i.
        U_cols = [ [U[j][i] for j in range(m)] for i in range(n)]
        additional = []
        for i in range(m):
            # Start with standard basis vector e_i.
            e = [1.0 if j == i else 0.0 for j in range(m)]
            # Subtract projections onto already computed columns.
            for u in U_cols + additional:
                proj = sum(e[k] * u[k] for k in range(m))
                e = [e[k] - proj * u[k] for k in range(m)]
            norm_e = norm(e)
            if norm_e > 1e-12:
                e = [x / norm_e for x in e]
                additional.append(e)
            if len(U_cols) + len(additional) == m:
                break
        # Append the additional columns to U_cols.
        U_cols.extend(additional)
        # Now transpose back to get U as an m x m matrix.
        U = [[U_cols[j][i] for j in range(m)] for i in range(m)]
    else:
        # If m == n, then U is already square.
        # Transpose the list of computed u_i (which are rows) into columns.
        U = [[U[j][i] for j in range(n)] for i in range(m)]
    
    # Step 5: Build S as an m x n diagonal matrix.
    S = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(min(m, n)):
        S[i][i] = singular_values[i]
    
    # Step 6: Form V^T from V.
    # Here, V is a list of n vectors of length n.
    V_T = transpose(V)
    
    return U, singular_values, V_T



def compose_matrix(corrsponding_points):
    A = []
    for (X, Y, Z), (u, v) in corrsponding_points:
        row1 = [
            -X, -Y, -Z, -1,
             0,  0,  0,  0,
             u*X, u*Y, u*Z, u
        ]
        row2 = [
             0,  0,  0,  0,
            -X, -Y, -Z, -1,
             v*X, v*Y, v*Z, v
        ]
        A.append(row1)
        A.append(row2)
    return A

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
    U, S, VT = np.linalg.svd(A)
    h = VT[-1, :]
    H = h.reshape(3,3)
    # Normalize so that H[2,2] = 1
    if abs(H[2,2]) > 1e-12:
        H = H / H[2,2]
    return H

def compute_vij(H, i, j):
    """
    For a given homography H (3x3 NumPy array), compute the vector v_ij as defined in Zhang’s paper.
    Here we follow:
      v_ij = [ h_i1*h_j1,
               h_i1*h_j2 + h_i2*h_j1,
               h_i2*h_j2,
               h_i3*h_j1 + h_i1*h_j3,
               h_i3*h_j2 + h_i2*h_j3,
               h_i3*h_j3 ]
    where h_i denotes the i-th row of H.
    Note: Many formulations use the columns; here we assume H's columns are h1, h2, h3.
    """
    # We choose to extract columns for consistency:
    h1 = H[:,0]
    h2 = H[:,1]
    h3 = H[:,2]
    if i == 1 and j == 2:
        return np.array([h1[0]*h2[0],
                         h1[0]*h2[1] + h1[1]*h2[0],
                         h1[1]*h2[1],
                         h1[2]*h2[0] + h1[0]*h2[2],
                         h1[2]*h2[1] + h1[1]*h2[2],
                         h1[2]*h2[2]], dtype=np.float64)
    elif i == 1 and j == 1:
        return np.array([h1[0]*h1[0],
                         h1[0]*h1[1] + h1[1]*h1[0],
                         h1[1]*h1[1],
                         h1[2]*h1[0] + h1[0]*h1[2],
                         h1[2]*h1[1] + h1[1]*h1[2],
                         h1[2]*h1[2]], dtype=np.float64)
    elif i == 2 and j == 2:
        return np.array([h2[0]*h2[0],
                         h2[0]*h2[1] + h2[1]*h2[0],
                         h2[1]*h2[1],
                         h2[2]*h2[0] + h2[0]*h2[2],
                         h2[2]*h2[1] + h2[1]*h2[2],
                         h2[2]*h2[2]], dtype=np.float64)
    else:
        raise ValueError("Unsupported indices in compute_vij")

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
    U, S, VT = np.linalg.svd(V)
    b = VT[-1, :]  # solution corresponding to smallest singular value.
    # b corresponds to the symmetric matrix B = [B11, B12, B22, B13, B23, B33]
    B11, B12, B22, B13, B23, B33 = b
    # Recover intrinsic parameters from B using the formulas from Zhang's paper.
    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lambda_val = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    if lambda_val < 0:
        print("Negative lambda encountered. Calibration failed.")
        return None, None
    alpha = math.sqrt(lambda_val / B11)
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

def main():
    # -- Load calibration checkboard images -- #
    calib_dir = "calibration_images"  # Directory with calibration images
    board_size = (11, 7)  # Number of inner corners in the chessboard pattern
    square_size = 0.030  # Size of a square in meters (e.g., 2.5 cm)
    K, extrinsics = multi_image_calibration(calib_dir, board_size, square_size)
    if K is None or extrinsics is None:
        print("Calibration failed.")
        return
    
    print("Extrinsics")
    for i, (R, t) in enumerate(extrinsics):
        print(f"Image {i}:")
        print("Rotation Matrix R:")
        print(R)
        print("Translation Vector t:")
        print(t)

    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    U, S, V_T = svd(A)
    U = np.array(U)
    S = np.array(S)
    V_T = np.array(V_T)
    # U, S, V_T = np.linalg.svd(A)
    # print("U:", U)
    # print("Singular Values:", S)
    # print("V^T:", V_T)

    # try to multiply U, S, V_T to get back the original matrix
    recontructed_A = np.dot(np.dot(U, np.diag(S)), V_T)
    print("Reconstructed A:", recontructed_A)

if __name__ == "__main__":
    main()
