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
    def __init__(self, env, draw_payload=False):
        self.vis = meshcat.Visualizer()
        self.vis["/Cameras/default"].set_transform(
            tf.translation_matrix([0.5, 0, 0]).dot(
                tf.euler_matrix(np.radians(-40), np.radians(0), np.radians(-100))))
        self.vis["/Cameras/default/rotated/<object>"].set_transform(
            tf.translation_matrix([-2, 0, 2.5]))
        self._setObstacles(env["environment"]["obstacles"])
        self.env = env
        self.draw_payload = draw_payload
        state_start = self.env["robots"][0]["start"]
        self.nb_bodies = int(len(state_start)/13)

        self._addQuadsPayload("start", "green")
        self.updateVis(state_start, "start")
        state_goal = self.env["robots"][0]["goal"]
        self._addQuadsPayload("goal", "red")
        self.updateVis(state_goal, "goal")

    def draw_traces(self, result, desired):
        if desired: 
            d = "_d"
            c_payload = 0xff0022
            c_quad = 0xff0022
        else:
            d = ""
            c_payload = 0xff1111  # a lighter red
            c_quad = 0x0000ff    # blue
        counter = 1
        if self.draw_payload:
            # trace payload:
            payload = result[:, :3].T
            self.vis["trace_payload"+d].set_object(
                g.Line(g.PointsGeometry(payload), g.LineBasicMaterial(color=c_payload)))
        else:
            counter = 0
        for i in range(counter, self.nb_bodies):
            quad_pos = result[:, 7*i: 7*i + 3].T
            self.vis["trace_quad" + str(i) + d].set_object(
                g.Line(g.PointsGeometry(quad_pos), g.LineBasicMaterial(color=c_quad)))

    def _addQuadsPayload(self, prefix: str = "", color_name: str = ""):
        counter = 1
        if self.draw_payload:
            self.vis[prefix + "_payload"].set_object(g.Mesh(
                g.Sphere(0.05), g.MeshLambertMaterial(DnametoColor.get(color_name, 0xff11dd))))
        else:
            counter = 0

        for i in range(counter, self.nb_bodies):
            self.vis[prefix + "_quad_" + str(i)].set_object(g.StlMeshGeometry.from_file(
                Path(__file__).parent.parent / 'meshes/cf2_assembly.stl'), g.MeshLambertMaterial(color=DnametoColor.get(color_name, 0xffffff)))
            self.vis[prefix + "_cable_" + str(i)].set_object(g.Box([0.0025,0.0025,1.0]), g.MeshLambertMaterial(color=0x000000))
            self.vis[prefix + "_sphere_" + str(i)].set_object(
                g.Mesh(g.Sphere(0.1), g.MeshLambertMaterial(opacity=0.1)))  # safety distance

    def _setObstacles(self, obstacles):
        for idx, obstacle in enumerate(obstacles):
            obsMat = g.MeshLambertMaterial(opacity=0.5, color=0x008000)
            center = obstacle["center"]
            shape = obstacle["type"]
            if (shape == "sphere"):
                radius = obstacle["radius"]
                self.vis["obstacle" +
                         str(idx)].set_object(g.Mesh(g.Sphere(radius), material=obsMat))
                self.vis["obstacle" +
                         str(idx)].set_transform(tf.translation_matrix(center))
            elif shape == "box":
                size = obstacle["size"]
                self.vis["obstacle" +
                         str(idx)].set_object(g.Mesh(g.Box(size), material=obsMat))
                self.vis["obstacle" +
                         str(idx)].set_transform(tf.translation_matrix(center))

    def updateVis(self, state, prefix: str = "", frame=None):
        # color of the payload trajectory
        point_color = np.array([1.0, 1.0, 1.0])
        counter = 1
        if self.draw_payload:
            payloadSt = np.array(state,dtype=np.float64)[0:7].T
            payload_pos = payloadSt[0:3]
            payload_quat = payloadSt[3:7]
        else:
            counter = 0
        if frame is not None:
            if self.draw_payload:
                frame[prefix + '_payload'].set_transform(
                    tf.translation_matrix(payload_pos).dot(
                        tf.quaternion_matrix(payload_quat)))

            for i in range(counter,self.nb_bodies):
                quad_st = state[7*i: 7*i + 7]
                quad_pos = quad_st[0:3]
                qw = quad_st[6]
                qxyz = quad_st[3:6]
                quad_quat = [qw, qxyz[0], qxyz[1], qxyz[2]]
                frame[prefix + "_quad_" + str(i)].set_transform(
                    tf.translation_matrix(quad_pos).dot(
                        tf.quaternion_matrix(quad_quat)))
                
                if self.draw_payload:
                    l = np.linalg.norm(payload_pos - quad_pos)
                    qc = (payload_pos - quad_pos)/l
                    cablePos  = payload_pos - l*np.array(qc)/2
                    cableQuat = rn.vector_vector_rotation(qc, [0,0,-1])
                    T = tf.identity_matrix()
                    T[0:3,3] = cablePos
                    R = tf.quaternion_matrix(cableQuat)
                    T[0:3,0:3] = R[0:3,0:3]


                    scale_factors = [1.0, 1.0, l]

                    frame[prefix + "_cable_" + str(i)].set_property(
                        "scale",         # prop name
                        "vector3",       # jstype: JS vec3
                        scale_factors    # value: list of 3 floats
                    )
                    frame[prefix + "_cable_" + str(i)].set_transform(T)
                frame[prefix + "_sphere_" + str(i)].set_transform(tf.translation_matrix(quad_pos))
        else:
            counter = 1
            if self.draw_payload:
                self.vis[prefix + '_payload'].set_transform(
                    tf.translation_matrix(payload_pos).dot(
                        tf.quaternion_matrix(payload_quat)))
            else:
                counter = 0
            for i in range(counter,self.nb_bodies):
                quad_st = state[7*i: 7*i + 7]
                quad_pos = quad_st[0:3]
                qw = quad_st[6]
                qxyz = quad_st[3:6]
                quad_quat = [qw, qxyz[0], qxyz[1], qxyz[2]]
                self.vis[prefix + "_quad_" + str(i)].set_transform(
                    tf.translation_matrix(quad_pos).dot(
                        tf.quaternion_matrix(quad_quat)))
                
                if self.draw_payload:
                    l = np.linalg.norm(payload_pos - quad_pos)
                    qc = (payload_pos - quad_pos)/l
                    cablePos  = payload_pos - l*np.array(qc)/2
                    cableQuat = rn.vector_vector_rotation(qc, [0,0,-1])
                    T = tf.identity_matrix()
                    T[0:3,3] = cablePos
                    R = tf.quaternion_matrix(cableQuat)
                    T[0:3,0:3] = R[0:3,0:3]

                    self.vis[prefix + "_cable_" + str(i)].set_transform(T.dot(tf.scale_matrix(l, [0,0,0], [0,0,-1])))
                self.vis[prefix + "_sphere_" + str(i)].set_transform(tf.translation_matrix(quad_pos))






def mujoco_meshcat():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help="environment")
    parser.add_argument('--ref', type=str, help="reference trajectory")
    parser.add_argument('--result', type=str, help="result trajectory")
    parser.add_argument('--output', type=str, help="output file")
    parser.add_argument('--payload', action='store_false', help="Show payload in visualization")
    args = parser.parse_args()
    pathtoenv = args.env
    with open(pathtoenv, "r") as file:
        env = yaml.safe_load(file)

    start = env["robots"][0]["start"]
    goal = env["robots"][0]["goal"]
    obstacles = env["environment"]["obstacles"]

    if args.result is not None:
        with open(args.result, 'r') as file:
            __path = yaml.safe_load(file)

        if "states" in __path and "actions" in __path:
            __path = __path
        else:
            __path = __path["result"]
        vis__ = False
        
        if "states" in __path and "actions" in __path:
            vis__ = True
        

    visualizer = Visualizer(env, args.payload)
    pathtoresult = args.result
    if not vis__:
        pathtoresult = pathtoresult.replace(".trajopt.yaml", "")
    if args.result is not None:

        with open(pathtoresult, 'r') as file:
            path = yaml.safe_load(file)

        if "states" in path:
            states = path['states']
        elif "result" in path:
            if vis__:
                states = path['result']['states']
                actions = path['result']['actions']
            else:
                states = path['result'][0]['states']
                actions = path['result'][0]['actions']
        else: 
            raise NotImplementedError("unknown result format")
        
        visualizer._addQuadsPayload()
        
        if args.ref is not None: 
            with open(args.ref, 'r') as file: 
                refpath = yaml.safe_load(file)
            if "states" in refpath:
                states_d = refpath["states"]
            elif "result" in refpath:
                if isinstance(refpath["result"], dict):
                    states_d = refpath["result"]["states"]
                elif isinstance(refpath["result"], list):
                    states_d = refpath["result"][0]["states"]
            else: 
                raise NotImplementedError("unknown result format")

            desired = True
            visualizer.draw_traces(np.array(states_d, dtype=np.float64), desired)
        desired = False
        visualizer.draw_traces(np.array(states, dtype=np.float64), desired)

        anim = Animation(default_framerate=1/0.02)  # 50 Hz
        for k, state in enumerate(states):
            with anim.at_frame(visualizer.vis, k) as frame:
                visualizer.updateVis(np.array(state, dtype=np.float64), frame=frame)
                
        visualizer.vis.set_animation(anim)

        res = visualizer.vis.static_html()
        # save to a file
        with open(args.output, "w") as f:
            f.write(res)


if __name__ == "__main__":
    mujoco_meshcat()