import math
import numpy as np
import rowan as rn
import yaml
import time
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from meshcat.animation import convert_frames_to_video


DnametoColor = {
    "red": 0xff0000,
    "green": 0x00ff00,
    "blue": 0x0000ff,
    "yellow": 0xffff00,
    "white": 0xffffff,
}


class Visualizer():
    def __init__(self, num_prims=1):
        self.vis = meshcat.Visualizer()
        self.vis["/Cameras/default"].set_transform(
            tf.translation_matrix([0.5, 0, 0]).dot(
                tf.euler_matrix(np.radians(-40), np.radians(0), np.radians(-100))))
        self.vis["/Cameras/default/rotated/<object>"].set_transform(
            tf.translation_matrix([-2, 0, 2.5]))
        self.nb_bodies = num_prims
        # self._addQuad()

    def draw_traces(self, prims_dict):
        c_quad = 0x0000ff    # blue
        for i in range(len(prims_dict)):
            state = prims_dict[i]
            s = np.array(state, dtype=np.float64)
            quad_pos = s[:, 0: 3].T
            self.vis["trace_quad" + str(i)].set_object(
                g.Line(g.PointsGeometry(quad_pos), g.LineBasicMaterial(color=c_quad)))

    def _addQuad(self, prefix: str = "", color_name: str = ""):
        print("Adding quad visualizer: ", self.nb_bodies)
        for i in range(0, self.nb_bodies):
            self.vis[prefix + "_quad_" + str(i)].set_object(g.StlMeshGeometry.from_file(
                Path(__file__).parent.parent / 'meshes/cf2_assembly.stl'), g.MeshLambertMaterial(color=DnametoColor.get(color_name, 0xffffff)))
            self.vis[prefix + "_sphere_" + str(i)].set_object(
                g.Mesh(g.Sphere(0.1), g.MeshLambertMaterial(opacity=0.1)))  # safety distance



    def updateVis(self, states, prefix: str = "", frame=None):
        point_color = np.array([1.0, 1.0, 1.0])
        if frame is not None:
            for i in range(0,self.nb_bodies):
                state = states[i]
                quad_st = state
                quad_pos = quad_st[0:3]
                quad_quat = quad_st[3:7]
                frame[prefix + "_quad_" + str(i)].set_transform(
                    tf.translation_matrix(quad_pos).dot(
                        tf.quaternion_matrix(quad_quat)))
                frame[prefix + "_sphere_" + str(i)].set_transform(tf.translation_matrix(quad_pos))
                    

        else:
            for i in range(0,self.nb_bodies):
                state = states[i]
                quad_st = state
                quad_pos = quad_st[0:3]
                quad_quat = quad_st[3:7]
                self.vis[prefix + "_quad_" + str(i)].set_transform(
                    tf.translation_matrix(quad_pos).dot(
                    tf.quaternion_matrix(quad_quat)))
                
                self.vis[prefix + "_sphere_" + str(i)].set_transform(tf.translation_matrix(quad_pos))

def prims_meschat():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prims', type=str, help="yaml prims file")
    parser.add_argument('--output', type=str, default="../output.html", help="output html file")
    args = parser.parse_args()


    if args.prims is not None:
        print("Loading primitives from: ", args.prims)
        with open(args.prims, 'r') as file:
            prims_dict = yaml.load(file, Loader=yaml.CLoader)
        print("Loaded primitives from: ", args.prims)

    print("number of primitives: ", len(prims_dict))
    visualizer = Visualizer(num_prims=len(prims_dict))

    prims = []
    lengths = []
    for i in range(len(prims_dict)):
        states = prims_dict[i]["states"]
        prims.append(states)
        lengths.append(len(states))
    max_length = max(lengths)
    print("Longest primitive length:", max_length)
    # Pad each primitive to max_length by repeating last state
    for i in range(len(prims)):
        while len(prims[i]) < max_length:
            prims[i].append(prims[i][-1].copy())
    print("All primitives padded to length:", max_length)

    # Displace each primitive in x and y between limits (-3, 3)
    prims_with_displacement = []
    n_prims = len(prims)
    x_limits = (-5, 5)
    y_limits = (-5, 5)
    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    for i, prim in enumerate(prims):
        states = []
        # Random displacement in square
        rand_x = rng.uniform(x_limits[0], x_limits[1])
        rand_y = rng.uniform(y_limits[0], y_limits[1])
        displacement = np.array([rand_x, rand_y, 0.0])
        for state in prim:
            new_state = state.copy()
            new_state[0:3] += displacement
            states.append(new_state)
        prims_with_displacement.append(states)
    prims = prims_with_displacement

    visualizer.draw_traces(prims)
    # Stack states by timestep: states[t] = [prims[0][t], prims[1][t], ...]
    states = []
    for t in range(max_length):
        states_at_t = [prims[p][t] for p in range(len(prims))]
        states.append(states_at_t)
    


    anim = Animation(default_framerate=1/0.1)  # 10 Hz
    for j ,state in enumerate(states):
        with anim.at_frame(visualizer.vis, j) as frame:
            pass
            # visualizer.updateVis(state, frame=frame)
            
    visualizer.vis.set_animation(anim)

    res = visualizer.vis.static_html()
    # save to a file
    with open(args.output, "w") as f:
        f.write(res)


if __name__ == "__main__":
    prims_meschat()