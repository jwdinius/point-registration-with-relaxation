import numpy as np
import json

from pyregistration import PythonRegistration, PythonConfig

def indices2Correspondences(inds, num_src, num_tgt):
    return np.array([(ind // num_tgt, ind % num_tgt) for ind in inds[:num_src]])

if __name__ == "__main__":
    print("Making test data...")
    m, n = 15, 30

    np.random.seed(seed=11011)
    # sample on square [-1, 1] x [-1, 1]
    # use homogeneous coordinates
    target_pts = 2. * np.random.random((4, n)) - 1.
    target_pts[2, :] = 0.
    target_pts[3, :] = 1.

    # transform
    ang = np.pi / 4.
    ca, sa = np.cos(ang), np.sin(ang)
    xt, yt = 1.0, 2.0
    tgt_to_src = np.array([[ca, sa, 0., xt],
                           [-sa, ca, 0., yt],
                           [0., 0., 1., 0.],
                           [0., 0, 0., 1.]])
    # map target points and then subsample (m < n)
    target_pts_xform = np.dot(tgt_to_src, target_pts)
    correspondences = np.random.choice(n, m, replace=False)
    #source_pts = target_pts_xform[:, correspondences] + 0.02*np.random.randn(4, m)
    source_pts = target_pts_xform[:, correspondences]
    source_pts[2, :] = 0.
    source_pts[3, :] = 1.

    # for test validation
    src_to_tgt = np.linalg.inv(tgt_to_src)
    source_pts_xform = np.dot(src_to_tgt, source_pts)
    data = {"source_pts": source_pts[:3, :].tolist(),
            "target_pts": target_pts[:3, :].tolist(),
            "correspondences": correspondences.tolist(),
            "src_to_tgt": src_to_tgt.tolist()}

    print("correspondences: ")
    print(np.array(data['correspondences']))

    pc = PythonConfig()
    pc.epsilon = 0.1
    pc.pairwise_dist_threshold = 0.2

    pr = PythonRegistration(source_pts[:3, :].tolist(), target_pts[:3, :].tolist(), pc)
    z_out = np.array(pr.findOptimumVector())

    indices = np.argwhere(z_out > 0.5)
    correspondences = indices2Correspondences(indices, m, n)
    tgt_indices = np.array([b for _, b in correspondences]).flatten()
    print("found correspondences: ")
    print(tgt_indices)

