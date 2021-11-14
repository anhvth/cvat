#!/usr/bin/env python3
#
# SPDX-License-Identifier: MIT
import logging
import os
import sys
from glob import glob
from http.client import HTTPConnection

import requests
from core.core import CLI, CVAT_API_V1
from core.definition import ResourceType
# from dotenv import load_dotenv
from env import *

log = logging.getLogger(__name__)


# load_dotenv()
SERVER_HOST = CVAT_SERVER_HOST
SERVER_PORT = CVAT_PORT
AUTH = CVAT_AUTH


def config_log(level):
    log = logging.getLogger("core")
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(level)
    if level <= logging.DEBUG:
        HTTPConnection.debuglevel = 1


config_log(20)


def create_task(project_id, task_name, path_imgs, path_anno):
    assert isinstance(path_imgs, list) is True, "path_imgs must be list of paths to image"
    assert isinstance(path_anno, str) is True, "path_anno must be str to path of annotation"
    base(
        action="create",
        name=task_name,
        project_id=project_id,
        resource_type=ResourceType.LOCAL,
        resources=path_imgs,
        annotation_format="COCO 1.0",
        annotation_path=path_anno,
    )


def dump_task():
    pass


def base(action, **kwargs):


    actions = {
        "create": CLI.tasks_create,
        # "delete": CLI.tasks_delete,
        "ls": CLI.tasks_list,
        # "frames": CLI.tasks_frame,
        "dump": CLI.tasks_dump,
        # "upload": CLI.tasks_upload,
        # "export": CLI.tasks_export,
        # "import": CLI.tasks_import,
        "data": CLI.tasks_data,
    }

    if action not in actions:
        raise Exception("'action' mismatched")

    with requests.Session() as session:
        api = CVAT_API_V1("%s:%s" % (SERVER_HOST, SERVER_PORT), False)
        import ipdb; ipdb.set_trace()
        log.info(f"AUTH:{AUTH}")
        cli = CLI(session, api, AUTH)

        cli_args = dict()
        if action == "create":
            # cli_args["name"] = "data_20210427_163937_cam150_1619516450"
            # cli_args["project_id"] = 2
            # cli_args["resource_type"] = ResourceType.LOCAL
            # cli_args["resources"] = file_imgs
            # cli_args["annotation_format"] = "COCO 1.0"
            # cli_args["annotation_path"] = file_anno
            cli_args["name"] = kwargs["name"]
            cli_args["project_id"] = kwargs["project_id"]
            cli_args["resource_type"] = kwargs["resource_type"]
            cli_args["resources"] = kwargs["resources"]
            cli_args["annotation_format"] = kwargs["annotation_format"]
            cli_args["annotation_path"] = kwargs["annotation_path"]

            cli_args["labels"] = None
            cli_args["overlap"] = 0
            cli_args["segment_size"] = 0
            cli_args["bug"] = ""

        elif action == "data":
            cli_args["task_id"] = 20
            cli_args["resource_type"] = ResourceType.LOCAL
            cli_args["resources"] = file_imgs
        elif action == "ls":
            cli_args["use_json_output"] = True
        elif action == "upload":
            cli_args["task_id"] = 20
            cli_args["fileformat"] = "COCO 1.0"
            cli_args["filename"] = file_anno

        try:
            actions[action](cli, **cli_args)
        except (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException,
        ) as e:
            log.critical(e)


if __name__ == "__main__":
    main()
