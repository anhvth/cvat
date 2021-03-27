from mmseg.datasets import CityscapesDataset

import random
def get_color():
    return "#%06x" % random.randint(0, 0xFFFFFF)

CLASSES = CityscapesDataset.CLASSES
rt = []
for class_id, class_name in enumerate(CLASSES):
    rt.append(
  {
    "name": class_name,
    "id": class_id,
    "color": get_color(),
    "attributes": []
  },
    )

import json
with open("color.json", "w") as f:
    json.dump(rt, f)