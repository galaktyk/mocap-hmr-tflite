""" 
Tensorflow SMPL implementation as batch.
Specify joint types:
'coco': Returns COCO+ 19 joints
'lsp': Returns H3.6M-LSP 14 joints
Note: To get original smpl joints, use self.J_transformed
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle as pickle

from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation
import os


module_path = os.path.abspath(os.path.dirname(__file__))
PKL_PATH = os.path.join(module_path, "smpl_model.pkl")


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

def batch_orth_proj_idrot(X, camera):
    """
    X is N x num_points x 3
    camera is N x 3
    same as applying orth_proj_idrot to each N 
    """


    camera = np.reshape(camera, [-1, 1, 3])

    X_trans = X[:, :, :2] + camera[:, :, 1:]

    shape = np.shape(X_trans)
    return np.reshape(
        camera[:, :, 0] * np.reshape(X_trans, [shape[0], -1]), shape)
    
        
class SMPL(object):
    def __init__(self, pkl_path=PKL_PATH, joint_type='cocoplus', dtype=np.float32):
        """
        pkl_path is the path to a SMPL model
        """
        # -- Load SMPL params --
        with open(pkl_path, 'rb') as f:
            dd = pickle.load(f, encoding="latin-1") 

        # Mean template vertices
        self.v_template = np.array(
            undo_chumpy(dd['v_template']),           
            dtype=dtype)
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis: 6980 x 3 x 10
        # reshaped to 6980*30 x 10, transposed to 10x6980*3
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = np.array(
            shapedir, dtype=dtype)

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor = np.array(
            dd['J_regressor'].T.todense(),        
            dtype=dtype)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        num_pose_basis = dd['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = np.array(
            posedirs, dtype=dtype)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = np.array(
            undo_chumpy(dd['weights']),
            dtype=dtype)

        # This returns 19 keypoints: 6890 x 19
        self.joint_regressor = np.array(
            dd['cocoplus_regressor'].T.todense(),         
            dtype=dtype)
        if joint_type == 'lsp':  # 14 LSP joints!
            self.joint_regressor = self.joint_regressor[:, :14]

        if joint_type not in ['cocoplus', 'lsp']:
            print('BAD!! Unknown joint type: %s, it must be either "cocoplus" or "lsp"' % joint_type)
            import ipdb
            ipdb.set_trace()

    def __call__(self, beta, theta, get_skin=False):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6980 x 3
        """


        num_batch = beta.shape[0]

        # 1. Add shape blend shapes
        # (N x 10) x (10 x 6890*3) = N x 6890 x 3
        v_shaped = np.reshape(
            np.matmul(beta, self.shapedirs),
            [-1, self.size[0], self.size[1]]) + self.v_template

        # 2. Infer shape-dependent joint locations.
        Jx = np.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = np.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = np.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = np.stack([Jx, Jy, Jz], axis=2)

        # 3. Add pose blend shapes
        # N x 24 x 3 x 3
        Rs = np.reshape(
            batch_rodrigues(np.reshape(theta, [-1, 3])), [-1, 24, 3, 3])
     
        # Ignore global rotation.
        pose_feature = np.reshape(Rs[:, 1:, :, :] - np.eye(3),
                                    [-1, 207])

        # (N x 207) x (207, 20670) -> N x 6890 x 3
        v_posed = np.reshape(
            np.matmul(pose_feature, self.posedirs),
            [-1, self.size[0], self.size[1]]) + v_shaped

        #4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)

        # 5. Do skinning:
        # W is N x 6890 x 24
        W = np.reshape(
            np.tile(self.weights, [num_batch, 1]), [num_batch, -1, 24])
        # (N x 6890 x 24) x (N x 24 x 16)
        T = np.reshape(
            np.matmul(W, np.reshape(A, [num_batch, 24, 16])),
            [num_batch, -1, 4, 4])
        v_posed_homo = np.concatenate(
            [v_posed, np.ones([num_batch, v_posed.shape[1], 1])], 2)
        v_homo = np.matmul(T, np.expand_dims(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        # Get cocoplus or lsp joints:
        joint_x = np.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = np.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = np.matmul(verts[:, :, 2], self.joint_regressor)
        joints = np.stack([joint_x, joint_y, joint_z], axis=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints


    def get_details(self, theta):
        """
            purpose:
                convert 85 vector to verts, joint2d, joint3d, Rotation matrix

            inputs:
                theta: N X (3 + 72 + 10)

            return:
                thetas, verts, j2d, j3d, Rs
        """
        
        

        cam = theta[:, 0:3]
        pose = theta[:, 3:75]
        shape = theta[:, 75:]
        verts, j3d, rs = self.__call__(beta=shape, theta=pose, get_skin=True)
   
        j2d = batch_orth_proj_idrot(j3d, cam)

        details = {
            'theta': theta,
            'cam': cam,
            'pose': pose,
            'shape': shape,
            'verts': verts,
            'j2d': j2d,
            'j3d': j3d
        }
        return details


