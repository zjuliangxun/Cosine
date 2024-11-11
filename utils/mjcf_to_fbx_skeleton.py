from typing import List
import os
import bpy
import numpy as np
import xml.etree.ElementTree as ET

"""
TODO: 
- Do not consider quat rotations
- can not load mesh
- refine left-right bone mapping
- don't need to add auxilary bones 
- leaf nodes direction/else
"""


def strlist2np(s: str):
    return np.array([float(i) for i in s.split(" ")])


def np2strlist(a: np.array):
    return " ".join([str(i) for i in a])


def np2tuple(a: np.array):
    return tuple(a.tolist())


class Body:
    def __init__(self, body_elem, parent=None):
        self.name = body_elem.get("name")
        self.pos = strlist2np(body_elem.get("pos", "0 0 0"))
        self.children: List[Body] = []
        self.parent: Body = parent
        self.parent_name = parent.name if parent is not None else None
        self.bone_length = None
        self.geom_type = None

        # get skeleton length from geom info
        geom_info = body_elem.findall("geom")
        if len(geom_info) == 0:
            return
        assert len(geom_info) <= 1, f"Multiple geoms found for body {body_elem.get('name')}"
        ge = geom_info[0]
        self.geom_type = ge.get("type")
        self.size = strlist2np(ge.get("size"))

        if self.geom_type == "sphere":
            radius = self.size[0].item()  # no radius * 2
            self.bone_length = self.pos / np.linalg.norm(self.pos) * radius
        elif self.geom_type == "box":
            max_edge = self.size.max()
            self.bone_length = np.eye(3)[np.argmax(self.size)] * max_edge
            # self.bone_length = self.parent.bone_length * max_edge
        else:  # elif self.geom_type == "capsule" or self.geom_type == "cylinder":
            a = strlist2np(ge.get("fromto"))
            self.start_p = a[:3]
            self.end_p = a[3:]
            self.bone_length = self.end_p - self.start_p
        # else:
        #     raise NotImplementedError(f"Unsupported geom type: {self.geom_type}!")

    def is_root(self):
        return self.parent_name is None

    def is_leaf(self):
        return len(self.children) == 0

    def print(self, cnt=0):
        print(f"{self.name}: {cnt}")
        cnt += 1
        for child in self.children:
            cnt = child.print(cnt)
        return cnt

    def t2fbx(self, output_path: str):
        # clear scene
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="ARMATURE")
        bpy.ops.object.delete()

        # create armature, this code always add redundant 'Bone' to the root!
        # bpy.ops.object.armature_add()
        arm_data = bpy.data.armatures.new("MJCF_Armature")
        arm_object = bpy.data.objects.new("MJCF_Armature", arm_data)
        bpy.context.collection.objects.link(arm_object)
        bpy.context.view_layer.objects.active = arm_object
        arm_object.select_set(True)
        bpy.context.view_layer.objects.active = arm_object

        bpy.ops.object.mode_set(mode="EDIT")
        self.create_skeleton()
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True, add_leaf_nodes=False)

    def _draw_skeleton(self, name, head, tail, parent=None):
        edit_bones = bpy.context.object.data.edit_bones
        bone = edit_bones.new(name)
        bone.head = np2tuple(head)
        bone.tail = np2tuple(tail)
        if parent is not None:
            bone.parent = edit_bones[parent]
            bone.use_connect = True

    def create_skeleton(self, local_frame=np.array([0, 0, 0.0])):
        if len(self.children) > 1:
            # for example, pelvis has 3 adjcent hips, we build the bone from its origin, and to its children[0] in the xml
            self._draw_skeleton(self.name, self.pos + local_frame, 1.001 * self.pos + local_frame, self.parent_name)
            for body in self.children:
                aux_name = self.name + f"_{body.name}_Joint"
                self._draw_skeleton(aux_name, self.pos + local_frame, body.pos + self.pos + local_frame, self.name)
                body.parent_name = aux_name
        elif self.is_leaf():
            # add end bone. the tail is not bind to any other one, so has to assign a length manully
            self._draw_skeleton(
                self.name, self.pos + local_frame, self.pos + local_frame + self.bone_length, self.parent_name
            )
        else:
            self._draw_skeleton(
                self.name, self.pos + local_frame, self.children[0].pos + self.pos + local_frame, self.parent_name
            )

        # recursive
        if self.is_leaf():
            return
        else:
            local_frame += self.pos
            for child in self.children:
                child.create_skeleton(local_frame.copy())


def parse_mjcf(file_path) -> List[Body]:
    tree = ET.parse(file_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    def traverse_bodies(body_elem, parent=None):
        body_info = Body(body_elem, parent)
        for child in body_elem.findall("body"):
            child_info = traverse_bodies(child, parent=body_info)
            body_info.children.append(child_info)

        return body_info

    if worldbody is not None:
        body_tree = [traverse_bodies(body) for body in worldbody.findall("body")]
    else:
        body_tree = []

    return body_tree


import json


class RokokoMotionRetarget:
    def __init__(self, source_armature, target_armature):
        self.source_armature_name = source_armature
        self.target_armature_name = target_armature
        bpy.context.scene.rsl_retargeting_armature_source = bpy.data.objects[source_armature]
        bpy.context.scene.rsl_retargeting_armature_target = bpy.data.objects[target_armature]
        bpy.ops.rsl.build_bone_list()
        self.total_bones = len(bpy.context.scene.rsl_retargeting_bone_list)

    def export_to_json(self, filename):
        data = []
        for i, bone in enumerate(bpy.context.scene.rsl_retargeting_bone_list):
            data.append(
                {"source": bone.bone_name_source, "target": bone.bone_name_target, "rsl_retargeting_bone_list_id": i}
            )
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Data exported to {filename}")

    def import_from_json(self, filename):
        with open(filename, "r") as file:
            data = json.load(file)
        print(f"Data imported from {filename}")
        assert self.total_bones == len(data), f"Bone number mismatch: {self.total_bones} != {len(data)}"
        bpy.ops.rsl.build_bone_list()

        for map in data:
            if map["target"] == "":
                continue
            # i = map["rsl_retargeting_bone_list_id"]
            # map = self._add_symetric_side_mapping(map, i)
            # assert (
            #     bpy.context.scene.rsl_retargeting_bone_list[i].bone_name_source == map["source"]
            # ), f"Bone name mismatch: {bpy.context.scene.rsl_retargeting_bone_list[i].bone_name_source} != {map['source']}"
            # bpy.context.scene.rsl_retargeting_bone_list[i].bone_name_target = map["target"]
            for i in range(len(bpy.context.scene.rsl_retargeting_bone_list)):
                if bpy.context.scene.rsl_retargeting_bone_list[i].bone_name_source == map["source"]:
                    bpy.context.scene.rsl_retargeting_bone_list[i].bone_name_target = map["target"]
                    print(bpy.context.scene.rsl_retargeting_bone_list[i].bone_name_source, "   ", map["target"])
                    break
            else:
                raise ValueError(
                    f"Bone name mismatch: {bpy.context.scene.rsl_retargeting_bone_list[i].bone_name_source} != {map['source']}"
                )

        return data

    def _add_symetric_side_mapping(self, map, index):
        """Automatically add right-side mapping if the source or target contains '_L_' or starts with 'L_'."""
        # keys = [('_L_','L_'), ('_R_','R_')]
        # for L,S in keys:
        source, target = map["source"], map["target"]
        if target == "":
            return map
        if "_L_" in source:
            if target.startswith("R_") or "_R_" in target:
                map["target"] = target.replace("R_", "L_", 1)
            right_source = source.replace("_L_", "_R_")
            right_target = target.replace("L_", "R_")
        elif "_R_" in source:
            if target.startswith("L_") or "_L_" in target:
                map["target"] = target.replace("L_", "R_", 1)
            right_source = source.replace("_R_", "_L_")
            right_target = target.replace("R_", "L_")
        else:
            return map  # No left-side pattern found, skip

        # Find corresponding right-side bone and update target
        for j, bone in enumerate(bpy.context.scene.rsl_retargeting_bone_list):
            if bone.bone_name_source == right_source:
                bpy.context.scene.rsl_retargeting_bone_list[j].bone_name_target = right_target
                print(f"Added symetric-side mapping: {right_source} -> {right_target}")
                break
        return map

    def retarget(self):
        bpy.context.scene.rsl_retargeting_auto_scaling = True
        bpy.ops.rsl.retarget_animation()

    def print_bone_trees(self, filepath):
        def print_bone_hierarchy(bone, file, indent=0):
            """Recursively print bone names in a hierarchy tree format."""
            file.write(" " * indent + bone.name)
            if len(bone.children) > 0:
                file.write(":\n")
            else:
                file.write("\n")
            for child in bone.children:
                print_bone_hierarchy(child, file, indent + 4)

        source_armature = bpy.data.objects[self.source_armature_name]
        target_armature = bpy.data.objects[self.target_armature_name]
        with open(filepath, "w") as file:
            if source_armature and source_armature.type == "ARMATURE":
                for bone in source_armature.data.bones:
                    if bone.parent is None:
                        print_bone_hierarchy(bone, file)
            if target_armature and target_armature.type == "ARMATURE":
                for bone in target_armature.data.bones:
                    if bone.parent is None:
                        print_bone_hierarchy(bone, file)


def run():
    ROOT_PATH = "/home/lx/Gitprojects/Cosine/assets/smpl_phc/"
    mjcf_path = os.path.join(ROOT_PATH, "smpl_humanoid.xml")
    humanoid_path = os.path.join(ROOT_PATH, "smpl_humanoid.fbx")
    # body_hierarchy = parse_mjcf(mjcf_path)
    # body_hierarchy = body_hierarchy[0]
    # body_hierarchy.print() # print each body's id
    # body_hierarchy.t2fbx(humanoid_path)
    raw_motion_path = "/home/lx/Gitprojects/Cosine/motion_clips/dataset_raw/Running.fbx"
    # import humanoid
    # bpy.ops.import_scene.fbx(filepath=humanoid_path, use_custom_normals=True, ignore_leaf_bones=False)
    # # import motion clip, for raw motion from mixamo, you must ignore leaf bones
    # bpy.ops.import_scene.fbx(filepath=raw_motion_path, ignore_leaf_bones=True)

    mr = RokokoMotionRetarget("Armature", "MJCF_Armature")
    # mr.print_bone_trees(os.path.join(ROOT_PATH, "tree.yaml"))
    # mr.export_to_json(os.path.join(ROOT_PATH, "retarget_bone_map.json"))
    mr.import_from_json(os.path.join(ROOT_PATH, "retarget_bone_map.json"))
    mr.retarget()


run()

"""
Implement a graph data structure in python. It describes many contact points in space. A graph contains many nodes, each of which represents a robot skeleton contacts with it. 
Each node has the following attributes: 
1. 3-D position(with repect to a local frame W)
2. normal vector to the contact plane
3. the skeleton id(an Enum).
Each edge has the following attributes:
1.the order that the contact transition happens. For example, if the robot first contacts with the ground, then contacts with the wall, the edge from the ground node to the wall node has a lower order than the edge from the wall node to other node. 
2. the frame number of each transition in the animation
3. 

Now build a Graph composing the above nodes and edges.It can :
1. Be serialized to a certain format
2. Record the frame W, and change the node positions if the W changes
3. Add a new node/edge to the graph
4. Manage all the contact transitions. Know how many transitions are there and return the frame number range of each transition
"""
