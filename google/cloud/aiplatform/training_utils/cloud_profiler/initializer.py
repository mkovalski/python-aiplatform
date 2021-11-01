# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import threading
from typing import Callable
from werkzeug import serving

from google.cloud.aiplatform.training_utils.cloud_profiler import webserver
from google.cloud.aiplatform.training_utils.cloud_profiler.plugins import base_plugin
from google.cloud.aiplatform.training_utils.cloud_profiler.plugins.tensorflow import (
    tf_profiler,
)


_AVAILABLE_PLUGINS = {"tensorflow": tf_profiler.TFProfiler}
_HOST_PORT = 6010


def _build_plugin(
    plugin: Callable[[], base_plugin.BasePlugin]
) -> base_plugin.BasePlugin:
    """Builds the plugin given the object.

    Args:
        plugin (Callable[[], base_plugin):
            Required. An uninitialized plugin.

    Returns:
        An initialized plugin.
    """
    if not plugin.can_initialize():
        logging.warning("Cannot initialize the plugin")
        return

    plugin.setup()

    if not plugin.post_setup_check():
        return

    return plugin()


def _run_app_thread(server: webserver.WebServer):
    """Run the webserver in a separate thread.

    Args:
        server (webserver.WebServer):
            Required. A webserver to accept requests.
    """
    daemon = threading.Thread(
        name="profile_server",
        target=serving.run_simple,
        args=("0.0.0.0", _HOST_PORT, server,),
    )
    daemon.setDaemon(True)
    daemon.start()


def initialize(plugin: str = "tensorflow"):
    """Initializes the profiling SDK.

    Args:
        plugin (str):
            Required. Name of the plugin to initialize.
            Current options are ["tensorflow"]

    Raises:
        ValueError:
            The plugin does not exist.
    """

    plugin_obj = _AVAILABLE_PLUGINS.get(plugin)

    if not plugin_obj:
        raise ValueError(
            "Plugin {} not available, must choose from {}".format(
                plugin, _AVAILABLE_PLUGINS.keys()
            )
        )

    prof_plugin = _build_plugin(plugin_obj)

    if not prof_plugin:
        return

    server = webserver.WebServer([prof_plugin])
    _run_app_thread(server)
