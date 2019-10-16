import numpy as np
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Making test data...")
    m, n = 15, 60

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
    source_pts = target_pts_xform[:, np.random.choice(n, m)] + 0.02*np.random.randn(4, m)
    source_pts[2, :] = 0.
    source_pts[3, :] = 1.

    # for test validation
    src_to_tgt = np.linalg.inv(tgt_to_src)
    source_pts_xform = np.dot(src_to_tgt, source_pts)
    data = {"source_pts": source_pts[:3, :].tolist(),
            "target_pts": target_pts[:3, :].tolist(),
            "src_to_tgt": src_to_tgt.tolist()}

    with open('../test/data.json', 'w') as json_file:
        json.dump(data, json_file)

    plt.subplot(221)
    plt.plot(target_pts[0, :], target_pts[1, :], '.')
    plt.subplot(222)
    plt.plot(target_pts[0, :], target_pts[1, :], '.', source_pts[0, :], source_pts[1, :], 'r*')
    plt.subplot(224)
    plt.plot(target_pts[0, :], target_pts[1, :], '.', source_pts_xform[0, :], source_pts_xform[1, :], 'r*')
    plt.show()


