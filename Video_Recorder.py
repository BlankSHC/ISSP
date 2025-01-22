####################################################
# If ERROR: GLEW initialization error: Missing GL version appears
# Solution: 
# 1. Add the following line to .bashrc:
#    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
# 2. In the terminal, run:
#    %%%%%%  unset LD_PRELOAD  %%%%%%
#    Then execute python main.py
####################################################

import os
import torch
from torchvision.io import write_video
import numpy as np
import imageio

class VideoRecorder:
    def __init__(self, video_path):
        self.video_path = video_path
        if self.video_path is not None:
            if not os.path.exists(self.video_path):
                os.makedirs(self.video_path)

    def init(self, eval_env, env_name, size=256, fps=30, enabled=True):
        if enabled:
            # self.video_size = size
            # self.video_fps = fps
            self.frames = []
            if 'halfcheetah' in env_name or 'hopper' in env_name or 'walker2d' in env_name:
                eval_env.render(mode='rgb_array', width=512, height=512, camera_id=0)
            elif 'antmaze-umaze' in env_name:
                # Call env.render() to ensure the viewer is initialized correctly
                eval_env.render(mode='rgb_array', width=512, height=512)
                # Set the camera's target position to (4, 4, 0)
                eval_env.viewer.cam.lookat[0] = 4.0
                eval_env.viewer.cam.lookat[1] = 4.0
                eval_env.viewer.cam.lookat[2] = 0
                eval_env.viewer.cam.fixedcamid = -1  # Use free camera
                # Define camera parameters for overhead view
                eval_env.viewer.cam.elevation = -90  # Set elevation to -90 degrees for overhead view
                eval_env.viewer.cam.azimuth = 0      # Set the azimuth
                eval_env.viewer.cam.distance = 25    # Set camera distance to 25 units
            elif 'antmaze-medium' in env_name:
                # Call env.render() to ensure the viewer is initialized correctly
                eval_env.render(mode='rgb_array', width=512, height=512)
                # Set the camera's target position to (10, 10, 0)
                eval_env.viewer.cam.lookat[0] = 9.9
                eval_env.viewer.cam.lookat[1] = 9.9
                eval_env.viewer.cam.lookat[2] = 0
                eval_env.viewer.cam.fixedcamid = -1
                # Define camera parameters for overhead view
                eval_env.viewer.cam.elevation = -90
                eval_env.viewer.cam.azimuth = 0
                eval_env.viewer.cam.distance = 40
            elif 'antmaze-large' in env_name:
                # Call env.render() to ensure the viewer is initialized correctly
                eval_env.render(mode='rgb_array', width=400, height=512)
                # Set the camera's target position to (4, 4, 0)
                eval_env.viewer.cam.lookat[0] = 18.0
                eval_env.viewer.cam.lookat[1] = 12.0
                eval_env.viewer.cam.lookat[2] = 0
                eval_env.viewer.cam.fixedcamid = -1
                # Define camera parameters for overhead view
                eval_env.viewer.cam.elevation = -90
                eval_env.viewer.cam.azimuth = 0
                eval_env.viewer.cam.distance = 58

    def record(self, eval_env, env_name):
        if hasattr(self, 'frames'):
            if 'antmaze-large' in env_name:
                image = eval_env.render(mode='rgb_array', width=400, height=512)
            else:
                image = eval_env.render(mode='rgb_array', width=512, height=512)
            self.frames.append(image)

    def release_video(self, name, myfps):
        self.video_name = name
        if hasattr(self, 'frames'):
            if self.video_path is not None:
                video_path = os.path.join(self.video_path, self.video_name)
                imageio.mimsave(video_path, self.frames, fps=myfps)

    def release_gif(self, name, myduration, myloop):
        self.video_name = name
        if hasattr(self, 'frames'):
            if self.video_path is not None:
                video_path = os.path.join(self.video_path, self.video_name)
                imageio.mimsave(video_path, self.frames, 'gif', duration=myduration, loop=myloop)
