""" Util functions for SMPL
@@batch_skew
@@batch_rodrigues
@@batch_lrotmin
@@batch_global_rigid_transformation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np



def scatter_nd_numpy(indices, updates, shape):
    target = np.zeros(shape, dtype=updates.dtype)
    indices = tuple(indices.reshape(-1, indices.shape[-1]).T)
    updates = updates.ravel()
    np.add.at(target, indices, updates)
    return target

def batch_skew(vec, batch_size=None):
    """
    vec is N x 3, batch_size is int

    returns N x 3 x 3. Skew_sym version of each matrix.
    """

    if batch_size is None:
        batch_size = vec.shape.as_list()[0]
    col_inds = np.array([1, 2, 3, 5, 6, 7])
    indices = np.reshape(
        np.reshape(np.arange(0, batch_size) * 9, [-1, 1]) + col_inds,
        [-1, 1])
    updates = np.reshape(
        np.stack(
            [
                -vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1],
                vec[:, 0]
            ],
            axis=1), [-1])
    out_shape = [batch_size * 9]
    res = scatter_nd_numpy(indices, updates, out_shape)
    res = np.reshape(res, [batch_size, 3, 3])

    return res


def batch_rodrigues(theta, name=None):
    """
    Theta is N x 3
    """

    batch_size = theta.shape[0]

    # angle = np.norm(theta, axis=1)
    # r = np.expand_dims(np.div(theta, np.expand_dims(angle + 1e-8, -1)), -1)
    # angle = np.expand_dims(np.norm(theta, axis=1) + 1e-8, -1)
    angle = np.expand_dims(np.linalg.norm(theta + 1e-8, axis=1), -1)
    r = np.expand_dims(theta/angle, -1)

    angle = np.expand_dims(angle, -1)
    cos = np.cos(angle)
    sin = np.sin(angle)

    outer = np.matmul(r, r.transpose(0,2,1))

    eyes = np.tile(np.expand_dims(np.eye(3), 0), [batch_size, 1, 1])
    R = cos * eyes + (1 - cos) * outer + sin * batch_skew(r, batch_size=batch_size)
    return R


def batch_lrotmin(theta, name=None):
    """ NOTE: not used bc I want to reuse R and this is simple.
    Output of this is used to compute joint-to-pose blend shape mapping.
    Equation 9 in SMPL paper.


    Args:
      pose: `Tensor`, N x 72 vector holding the axis-angle rep of K joints.
            This includes the global rotation so K=24

    Returns
      diff_vec : `Tensor`: N x 207 rotation matrix of 23=(K-1) joints with identity subtracted.,
    """
 
    theta = theta[:, 3:]

    # N*23 x 3 x 3
    Rs = batch_rodrigues(np.reshape(theta, [-1, 3]))
    lrotmin = np.reshape(Rs - np.eye(3), [-1, 207])

    return lrotmin


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False):
    """
    Computes absolute joint locations given pose.

    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.

    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index

    Returns
      new_J : `Tensor`: N x 24 x 3 location of absolute joints
      A     : `Tensor`: N x 24 4 x 4 relative joint transformations for LBS.
    """

    N = Rs.shape[0]
    if rotate_base:
        print('Flipping the SMPL coordinate frame!!!!')
        rot_x = np.constant(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=Rs.dtype)
        rot_x = np.reshape(np.tile(rot_x, [N, 1]), [N, 3, 3])
        root_rotation = np.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]

    # Now Js is N x 24 x 3 x 1
    Js = np.expand_dims(Js, -1)

    def make_A(R, t, name=None):
        # Rs is N x 3 x 3, ts is N x 3 x 1
      
        R_homo = np.pad(R, [[0, 0], [0, 1], [0, 0]])
        t_homo = np.concatenate([t, np.ones([N, 1, 1])], 1)
        return np.concatenate([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = np.matmul(
            results[parent[i]], A_here)
        results.append(res_here)

    # 10 x 24 x 4 x 4
    results = np.stack(results, axis=1)

    new_J = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    Js_w0 = np.concatenate([Js, np.zeros([N, 24, 1, 1])], 2)
    init_bone = np.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = np.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
    A = results - init_bone

    return new_J, A
