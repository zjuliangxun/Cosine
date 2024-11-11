# Cosine
The contact-based character animation controller

horovodrun -np 8 python rlg_train.py --task <task_name> --horovod --headless


## Build Your Dataset
1. build fbx character from mjcf
   - check leaf bone's direction's orintation
   - transform to origin and z as up axis
   - set character height. You can use mujoco to help to decide the Pelvis's height
   - do not add leaf bones when exporting using UI
2. get raw fbx animation
   - download data from maximo; if "with skin" the blender might raise OverFlowException. The raw fbx contains leaf skeleton
   - clip the end(Note to delete extra keyframes. Just setting the end time point doesn't work.)
   - rotate and trans to origin(注意不要应用scale的变换不然动画会很怪，同时又一定要对齐方向不然retarget错误), then use `ctrl+A`to apply the transform
   - **BUG** : DO NOT export the transformed fbx, otherwise the retargeting will fail. This may be related to mixamo's animation format......
3. motion retargeting
   - use `utils/mjcf_to_fbx_skeleton.py` to retarget(When use blender to import do not ignore leaf nodes)
   - clip the animation
   - store the file to `data_preprocessed/motion`
4. build the retargeted animation into npy
   1. switch a py310 env, and use pip to install fbx sdk
   2. use `utils/fbx_to_npy.py` 
5. annotate the animation's contact graph
   - build the json files(under `data_preprocessed/cg_jsons`) which declare the contact point's frame idx
   - set BLENDER=True in the `utils/annotate_cg.py` and run in blender to calculate the contact's position in the file
   - set BLENDER=False and store the cg into pkl files under `data_preprocessed/contact_graph`
   - **NOTE**: idle's main_line must be assigned by hand