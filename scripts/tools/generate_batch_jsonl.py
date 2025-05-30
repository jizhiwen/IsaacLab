import os
import sys

import glob
import json
import random

prompt_pattern = """The scene depicts a robotic arm performing a precise block-stacking operation with {cube_description}
cubes in a {location}. The workspace is centered on a {table_material} table. On the table are three cubes. The robotic
 arm, featuring a sleek industrial design clad in smooth white plastic with metallic joints,
 extends from its base to manipulate the cubes. The entire scene is illuminated by professional studio lighting,
 creating soft shadows. A clutter of equipment can be seen in the background. The camera is fixed in place.
"""

cube_description = [
    "glass", "brightly colored plastic", "wood", "steel", "worn dull colored", "cheese",
    "polished brass", "matte black", "translucent acrylic", "rustic copper", "brushed aluminum",
    "painted ceramic", "textured rubber", "carbon fiber", "iridescent metal", "frosted crystal",
    "ice", "origami paper"
]
table_material = [
    "wood", "marble", "metal", "glass", "stone",
    "polished stainless steel", "concrete", "brushed aluminum", "tempered glass", 
    "granite", "composite resin", "industrial plastic", "carbon fiber"
]
location = [
    "cluttered workshop", "manufacturing facility", "university laboratory", "restaurant",
    "high-tech cleanroom", "industrial warehouse", "robotics research center", 
    "automated assembly line", "testing facility", "engineering lab", 
    "technology demonstration room", "quality control station"
]

def demo(dataset: str):
    if not os.path.isdir(dataset):
        exit(1)

    if not os.path.isdir(os.path.join(dataset, "meta")):
        exit(1)

    episodes_json = os.path.join(dataset, "meta", "episodes.jsonl")
    if not os.path.isfile(os.path.join(dataset, "meta", "episodes.jsonl")):
        exit(1)
    output = open("./output_batch.jsonl", "w")
    if not output:
        exit(1)

    with open(episodes_json, "r") as f:
        for line in f:
            colume = json.loads(line)
            episode_index = colume.get("episode_index", None)
            tasks = colume.get("tasks", None)
            if not episode_index:
                continue


            batch_jsonl = {}
            for video in glob.glob(f"**/observation.images.ego_view.normals.shaded_segmentation/episode_{str(episode_index).zfill(6)}.mp4", root_dir=dataset, recursive=True):
                batch_jsonl["visual_input"] = os.path.abspath(video)
                batch_jsonl["prompt"] = prompt_pattern.format(cube_description=random.choice(cube_description),
                                                              location=random.choice(location),
                                                              table_material=random.choice(table_material))
                batch_jsonl["control_overrides"] = {}
                batch_jsonl["control_overrides"]["edge"] = {"input_control": None}
                output.write(json.dumps(batch_jsonl) + "\n")


demo(sys.argv[1])(base)