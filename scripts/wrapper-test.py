import numpy as np
import json
import matplotlib.pyplot as plt
from pyregistration import PythonRegistration, PythonConfig

if __name__ == "__main__":
    m, n = 15, 40
    noise_val = 0.0
    make_ut_data = False
    run_optimization = True
    make_plots = True

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
    source_pts = target_pts_xform[:, correspondences] + noise_val*np.random.randn(4, m)
    source_pts[2, :] = 0.
    source_pts[3, :] = 1.
    
    if make_ut_data:
        src_to_tgt = np.linalg.inv(tgt_to_src)
        data = {"source_pts": source_pts[:3, :].tolist(),
                "target_pts": target_pts[:3, :].tolist(),
                "correspondences": correspondences.tolist(),
                "src_to_tgt": src_to_tgt.tolist()}

        with open('../tests/testdata/registration-data.json', 'w') as json_file:
            json.dump(data, json_file)

    if run_optimization:
        pc = PythonConfig()
        pc.epsilon = 0.1
        pc.pairwiseDistThreshold = 0.1
        pc.corrThreshold = 0.5
        pc.doWarmStart = True

        pr = PythonRegistration(source_pts[:3, :].tolist(), target_pts[:3, :].tolist(), pc, source_pts.shape[1])

        if make_plots:
            z_out = np.array(pr.optimum)
            H_out = np.array(pr.transform)
            source_pts_xform = np.dot(H_out, source_pts)
            
            plt.figure()
            plt.plot(z_out, '.')
            plt.xlabel("i*n+j, i in src and j in tgt")
            plt.ylabel("z_{i,j} (row-major)")
            plt.title("Optimal Solution from IPOPT")
            plt.figure()
            ax1 = plt.subplot(121)
            ax1.plot(target_pts[0, :], target_pts[1, :], 'o', source_pts[0, :], source_pts[1, :], 'r.')
            ax1.set_ylabel("y")
            ax1.set_title("Original Alignment")
            ax2 = plt.subplot(122)
            ax2.plot(target_pts[0, :], target_pts[1, :], 'o', label="target")
            ax2.plot(source_pts[0, :], source_pts[1, :], 'r.', label="source")
            legend_made = False
            for c in pr.correspondences.keys():
                if not legend_made:
                    ax2.plot([target_pts[0, c[1]], source_pts[0, c[0]]], [target_pts[1, c[1]], source_pts[1, c[0]]], 'b--', label="correspondence")
                    legend_made = True
                else:
                    ax2.plot([target_pts[0, c[1]], source_pts[0, c[0]]], [target_pts[1, c[1]], source_pts[1, c[0]]], 'b--')
            ax2.legend()
            ax2.set_title("Identified Correspondences")
            ax2.set_xlabel("x")
            
            plt.figure()
            plt.plot(target_pts[0, :], target_pts[1, :], 'o', label="target")
            plt.plot(source_pts_xform[0, :], source_pts_xform[1, :], 'r.', label="source")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.title("Source Points Realigned")
            plt.show()
