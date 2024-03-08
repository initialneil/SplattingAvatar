FLAME from IMAvatar
=====
- By Zhijing SHAO, 2024.3.6
- This is the IMAvatar/DECA version of FLAME
- Originally from: https://github.com/zhengyuf/IMavatar/tree/main/code/flame
- What's changed from normal FLAME:
  - There's a `factor=4` in the `flame.py`, making the output mesh 4 times larger
  - The input `full_pose` is [Nx15], which is a combination of different pose components
  - In a standard FLAME model, there is `pose_params`[Nx6], `neck_pose`[Nx3], `eye_pose`[Nx6]. To convert to `full_pose`:
    ```
    # [3] global orient
    # [3] neck
    # [3] jaw
    # [6] eye
    full_pose = torch.concat([pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=-1)
    ```
- The community should stop using this version of FLAME code

