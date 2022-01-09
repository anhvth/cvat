import json
import os
import pickle
from datetime import datetime

import numpy as np

# import torch
# from mmcv.ops import bbox_overlaps

IOU_THRES = 0.3


class Task:
    def __init__(self, shapes: list, tracks: list) -> None:
        self.shapes = dict()
        self.frame2shapes = dict()
        self.tracks = dict()
        self.cache_interpolated_shapes = set()
        self.cache_del_shape_ids = set()

        for shape in shapes:
            self.add_shape(shape)

    def add_shape(self, shape):
        self.shapes[shape["id"]] = {**shape.copy(), "removed": False}

        frame = shape["frame"]
        if frame not in self.frame2shapes:
            self.frame2shapes[frame] = [shape["id"]]
        else:
            self.frame2shapes[frame].append(shape["id"])

    def add_track(self, track):
        # with open("track.json", 'w+') as f:
        #     json.dump(track, f, indent=2)
        #     f.write(f"track keys: {track.keys()}")

        sorted_shapes = sorted(track["shapes"], key=lambda x: x["frame"])
        track["shapes"] = sorted_shapes

        interpolated = set()
        for shape in track["shapes"]:
            ## Điều kiện này dùng để kiểm tra xem shape này có phải interpolated shape không
            ## bằng cách đối chiếu dấu vết
            trace = sum(shape["points"])
            if trace in self.cache_interpolated_shapes:
                self.cache_interpolated_shapes.remove(trace)
                interpolated.add(shape["frame"])

        # NOTE: Tạm thời để interpolated là list, sau này cần chuyển về set
        self.tracks[track["id"]] = {**track, "interpolated": list(interpolated)}


class TrackUpdater:
    def __init__(self) -> None:
        self.tasks = dict()

    ## NOTE: Nên nghĩ cách dùng session hoặc cache của Django thay vì cách lưu file này
    def save_pickle(self, pk, data):
        with open(f"task_{pk}.pkl", "wb+") as f:
            pickle.dump(data, f)

    def load_pickle(self, pk):
        with open(f"task_{pk}.pkl", "rb") as f:
            data = pickle.load(f)

        return data

    def init_task(self, data, pk, logger):
        logger.error("== CALLED: init")
        # if pk not in self.tasks:
        # self.tasks[pk] = Task(data["shapes"].copy(), data["tracks"].copy())
        if not os.path.isfile(f"task_{pk}.pkl"):
            task = Task(data["shapes"].copy(), data["tracks"].copy())
            self.save_pickle(pk, task)

    def patch_update(self, data, pk, logger):
        ## TODO: Cập nhật shape đơn lẻ hoặc shape thuộc track
        return data

    def patch_create_supplement(self, data, pk, logger):
        ## Method này chỉ xử lí track các track mới tạo từ lệnh PATCH_create.

        ## Lưu ý hàm này sẽ chỉ bổ sung các shape còn thiếu bằng interpolation. Tất cả
        ## shapes được trả về bằng hàm này đều không có ID. ID sẽ được cập nhật sau đó
        ## bằng serializer và bằng hàm 'patch_create_update_id'

        logger.error("== CALLED: patch_create_supplement")
        # task: Task = self.tasks[pk]
        task = self.load_pickle(pk)

        with open("before_create_sup.json", "w+") as f:
            f.write("task:: \n")
            json.dump({"shapes": task.shapes, "tracks": task.tracks}, f, indent=2)
            f.write("data:: \n")
            json.dump(data, f, indent=2)

        ## Create track

        for track in data["tracks"]:
            shapes = sorted(track["shapes"].copy(), key=lambda x: x["frame"])

            # Remove shapes in the middle of track whose outside=True
            i = 0
            while i < len(shapes):
                if shapes[i]["outside"] == True:
                    if i == len(shapes) - 1:
                        break

                    del shapes[i]
                else:
                    i += 1

            mixed_shapes = []  ## contains original shapes and interpolated shapes
            ## biến dưới đâu chứa ID của các shape trước đây đứng một mình nhưng
            ## bây giờ đã được bổ sung vào một track nhờ kiểm tra IoU
            del_shape_ids = set()
            for i in range(len(shapes) - 1):
                cur_shape, next_shape = shapes[i], shapes[i + 1]

                mixed_shapes.append(cur_shape)

                if next_shape["frame"] - cur_shape["frame"] > 1:
                    ## Do interpolation here
                    for frame in range(cur_shape["frame"] + 1, next_shape["frame"]):
                        interpolated_shape = interpolate(cur_shape, next_shape, frame)

                        ## Check IoU of interpolated_shape with any shape in same frame
                        matching = False
                        if frame in task.frame2shapes.keys():
                            ## TODO: Optimize this to calculate overlap for all shapes in frame
                            for shape_id in task.frame2shapes[frame]:
                                shape = task.shapes[shape_id]
                                if not shape["removed"]:
                                    interpolated_shape, matching = find_iou_matching_shape(
                                        shape, interpolated_shape
                                    )

                                    if matching:
                                        del_shape_ids.add(shape_id)
                                        break

                        if not matching:
                            ## Đây là một thủ thuật nhằm đánh dấu shape nào là interpolate
                            ## Một chút randomness đủ nhỏ để không ảnh hưởng tới vị trí của bounding box,
                            ## nhưng đủ để khiến các giá trị của bounding box là unique, nên tổng của chúng cũng unique
                            task.cache_interpolated_shapes.add(sum(interpolated_shape["points"]))

                        ## Add newly created/found shape to list
                        mixed_shapes.append(interpolated_shape)

            mixed_shapes.append(next_shape)
            track["shapes"] = mixed_shapes

            task.cache_del_shape_ids = task.cache_del_shape_ids.union(del_shape_ids)
            for shape_id in del_shape_ids:
                task.shapes[shape_id]["removed"] = True

        with open("after_create_sup.json", "w+") as f:
            f.write("shapes in task:: \n")
            json.dump(task.shapes, f, indent=2)

        self.save_pickle(pk, task)

        return data

    def patch_create_update_id(self, data, pk, logger):
        ## Hàm này chỉ được phép gọi sau khi đã chạy hàm 'patch_create_supplement'
        ## Hàm này sẽ dùng kết quả trả về của serializer để cập nhật ID cho các shape mới (bao gồm cả những interpolated shapes)

        logger.error("== CALLED: patch_create_update_id")

        # task: Task = self.tasks[pk]
        task = self.load_pickle(pk)

        with open("before_update_id.json", "w+") as f:
            f.write("shapes in task:: \n")
            json.dump(task.shapes, f, indent=2)
            f.write("tracks in task:: \n")
            json.dump(task.tracks, f, indent=2)
            f.write("shapes in data:: \n")
            json.dump(data["shapes"], f, indent=2)
            f.write("tracks in data:: \n")
            json.dump(data["tracks"], f, indent=2)

        for shape in data["shapes"]:
            task.add_shape(shape)

        for track in data["tracks"]:
            task.add_track(track)
        assert len(task.cache_interpolated_shapes) == 0

        with open("after_update_id.json", "w+") as f:
            f.write("shapes in task:: \n")
            json.dump(task.shapes, f, indent=2)
            f.write("tracks in task:: \n")
            # print_tracks = task.tracks.copy()
            # for track in print_tracks:
            #     track['interpolated'] = list(track['interpolated'])
            json.dump(task.tracks, f, indent=2)

        self.save_pickle(pk, task)

    def patch_delete(self, data, pk, logger):
        ## Xử lí trường hợp xoá box hoặc toàn bộ track, không có trường hợp xoá một box ở giữa track
        # Các shapes hoặc tracks trong 'data' có sẵn ID, chỉ việc dựa vào đó mà xoá trong cache

        logger.error("== CALLED: patch_delete")

        # task: Task = self.tasks[pk]
        task = self.load_pickle(pk)
        with open("delete.json", "w+") as f:
            f.write("shapes in task:: \n")
            json.dump(task.shapes, f, indent=2)
            f.write("shapes in data:: \n")
            json.dump(data["shapes"], f, indent=2)

        for shape in data["shapes"]:
            task.frame2shapes[shape["frame"]].remove(shape["id"])
            del task.shapes[shape["id"]]

        ## Thêm vào 'data' những shape đã được đánh dấu lược bỏ trong quá trình
        ## kết nạp vào track mới nhờ IoU đồng thời xoá luôn shape đó khỏi task
        already_del_shape_ids = {
            shape["id"] for shape in data["shapes"]
        }  ## Đây là danh sách các shape id yêu cầu xoá từ frontend
        tobe_del_shape_ids = task.cache_del_shape_ids
        ## Lọc ra các shape id xuất hiện trong danh sách lược bỏ của task nhưng không xuất hiện
        ## trong danh sách yêu cầu xoá từ frontend
        tobe_del_shape_ids = tobe_del_shape_ids - already_del_shape_ids
        for del_shape_id in tobe_del_shape_ids:
            shape = task.shapes[del_shape_id]
            del shape["removed"]
            data["shapes"].append(shape)

            del task.shapes[del_shape_id]
        task.cache_del_shape_ids = set()

        for track in data["tracks"]:
            del task.tracks[track["id"]]

        self.save_pickle(pk, task)

        return data

    def del_task(self, pk):
        del self.tasks[pk]


def interpolate(shape1: dict, shape2: dict, frame: int):
    alpha = (frame - shape1["frame"]) / (shape2["frame"] - shape1["frame"])
    b1, b2 = np.array(shape1["points"]), np.array(shape2["points"])
    new_b = (1 - alpha) * b1 + alpha * b2
    ## the next line ensures each value in array new_b is unique,
    ## which requires for backing up interpolating box
    new_b = new_b + np.random.rand(4)

    new_shape = {
        "type": "rectangle",
        "occluded": False,
        "points": new_b.tolist(),
        "outside": False,
        "attributes": [],
        "frame": frame,
    }

    return new_shape


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def find_iou_matching_shape(checking_shape, interpolated_shape):
    ret_shape = interpolated_shape
    iou = bb_intersection_over_union(checking_shape["points"], interpolated_shape["points"])
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # ious = (
    #     bbox_overlaps(
    #         torch.tensor([checking_shape["points"]], device=device),
    #         torch.tensor([interpolated_shape["points"]], device=device),
    #     )
    #     .cpu()
    #     .flatten()
    #     .numpy()
    # )
    matching = False
    # if ious[0] > IOU_THRES:
    if iou > IOU_THRES:
        ## checking_shape overlaps interpolated_shape
        matching = True
        ret_shape = checking_shape.copy()
        ret_shape["outside"] = False
        for rmv_field in ["id", "z_order", "label_id", "group", "source", "removed"]:
            del ret_shape[rmv_field]

    return ret_shape, matching


def writeln(task, action, *args):
    cur_stime = datetime.now().strftime("%m-%d_%H-%M-%S")
    action = f"_{action}" if action != "" else ""
    with open(f"log_{cur_stime}_{task}{action}.log", "w+") as f:
        f.write(f"request arrived {task}{action} \n")
        for arg in args:
            f.write(arg + "\n")
            f.write("*" * 10 + "\n")
        f.write("*" * 40 + "\n")
