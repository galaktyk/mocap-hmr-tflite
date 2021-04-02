# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['run.py'],
             pathex=['D:\\jobs\\2021\\future_play\\mocap-hmr-tflite'],
             binaries=[],
             datas=[('hmr/model/HMR.tflite', 'hmr/model'), ('person_detector/model/lite_pose_detection.tflite', 'person_detector/model'), ('hmr/model/initial_theta.npy', 'hmr/model/'), ('np_smpl/smpl_model.pkl', 'np_smpl')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='run',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
