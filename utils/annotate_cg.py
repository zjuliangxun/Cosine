import os
import json
import argparse
import pickle
import sys

json_paths = [
    "/home/lx/Gitprojects/Cosine/motion_clips/dataset_processed/cg_jsons/walk_cg.json",
    "/home/lx/Gitprojects/Cosine/motion_clips/dataset_processed/cg_jsons/running_cg.json",
    "/home/lx/Gitprojects/Cosine/motion_clips/dataset_processed/cg_jsons/idle_cg.json",
]


def run(json_path, BLENDER):
    with open(json_path, "r") as json_file:
        cfg = json.load(json_file)

    if BLENDER:
        import bpy

        def bone_pos_generator(cfg):
            armature_name = cfg.get("armature_name", "Armature")
            armature = bpy.data.objects.get(armature_name)
            if armature is None:
                raise ValueError(f"未找到名称为 '{armature_name}' 的对象，请检查对象名称。")
            BONE_NAME_MAP = cfg.get("bone_name_map", None)

            for key, value in cfg.items():
                try:
                    frame = int(key)
                except:
                    continue
                assert bpy.context.scene.frame_start <= frame < bpy.context.scene.frame_end, f"帧数 {frame} 超出范围。"
                bpy.context.scene.frame_set(frame)

                for node in value.get("nodes", []):
                    bone_name = node.get("bone") if BONE_NAME_MAP is None else BONE_NAME_MAP[node.get("bone")]
                    if bone_name not in armature.pose.bones:
                        raise ValueError(f"未找到名称为 '{bone_name}' 的骨骼，请检查骨骼名称。")
                    else:
                        pose_bone = armature.pose.bones[bone_name]
                        # head_global_position == armature.matrix_world @ pose_bone.matrix @ pose_bone.location
                        if node.get("use_head_pos", False):
                            head_global_position = pose_bone.head
                        else:
                            head_global_position = pose_bone.tail

                        node["pos"] = list(head_global_position)
                        print(node["pos"])

            return cfg

        bpy.ops.import_scene.fbx(filepath=cfg["motion_clip_file"], ignore_leaf_bones=False)

        cfg = bone_pos_generator(cfg)
        with open(json_path, "w") as json_file:
            json.dump(cfg, json_file, ensure_ascii=False, indent=4)

        # clear the scene
        bpy.ops.object.get(cfg.get("armature_name", "Armature")).select_set(True)
        bpy.ops.object.delete(use_global=False, confirm=False)

    else:
        from data.contact_graph_base import CNode, CEdge
        from data.contact_graph import ContactGraph

        def bone_pos_generator(cfg):
            BONE_NAME_MAP = cfg.get("bone_name_map", None)
            BONE_ID_MAP = cfg.get("bone_id_map", None)
            order_count = 0
            nodes, edges, idlist = [], [], []

            for key, value in cfg.items():
                try:
                    frame = int(key)
                except:
                    continue
                for node in value.get("nodes", []):
                    bone_name = node.get("bone") if BONE_NAME_MAP is None else BONE_NAME_MAP[node.get("bone")]

                    # TODO 检查sustain time是1还是0； order_count是不是从0开始
                    nodes.append(
                        CNode(
                            pos=node["pos"],
                            normal=node.get("normal", [0, 0, 1]),
                            skeleton_id=BONE_ID_MAP.get(bone_name),
                            order=order_count,
                            sustain_time=node.get("sustain_time", 1),
                            start_time=0,
                            end_time=0,
                        )
                    )
                    idlist.append(node.get("id"))
                order_count += 1

                sorted_nodes = [node for _, node in sorted(zip(idlist, nodes), key=lambda pair: pair[0])]
                nodes[:] = sorted_nodes

                for edge in value.get("edges", []):
                    start_id = edge.get("start")
                    edges.append(
                        CEdge(
                            start_node=start_id,
                            end_node=edge.get("end"),
                            order=nodes[start_id].order,
                            start_frame=0,
                            end_frame=0,
                        )
                    )

            return nodes, edges

        out_filepath = cfg["output_file"]
        nodes, edges = bone_pos_generator(cfg)
        cg = ContactGraph(nodes, edges, skill_name=cfg["skill_name"], main_line=cfg.get("main_line", None))

        with open(out_filepath, "wb") as out_file:
            pickle.dump(cg, out_file)
        print(f"ContactGraph saved to {out_filepath}")


for json_path in json_paths:
    BLENDER = False
    run(json_path, BLENDER)
