#/bin/bash

docker cp utils/track_updater/TrackUpdater.py cvat:/home/django/utils/track_updater/TrackUpdater.py
docker cp cvat/apps/engine/views.py cvat:/home/django/cvat/apps/engine/views.py
docker restart cvat && docker logs -f cvat