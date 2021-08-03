# -*- coding: utf-8 -*-

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import threading
from typing import Optional, List
from werkzeug import serving


from google.cloud.aiplatform.training_utils.cloud_training_tools import web_server
from google.cloud.aiplatform.training_utils.cloud_training_tools.plugins import (
    base_plugin,
)
from google.cloud.aiplatform.training_utils.cloud_training_tools.plugins.tf_profiler import (
    tf_profiler,
)

# Add any additional plugins here.
ALL_PLUGINS = {tf_profiler.TFProfiler.PLUGIN_NAME: tf_profiler.TFProfiler}


def _run_webserver(plugins, port):
    """Run the webserver that hosts the various cloud_training_tools plugins."""
    app = web_server.create_web_server(plugins)
    serving.run_simple("0.0.0.0", port, app)


def _run_app_thread(plugins: List[base_plugin.BasePlugin], port: int) -> None:
    """Run the cloud_training_tools web server in a separate thread."""
    daemon = threading.Thread(
        name="cloud_training_tools_server", target=_run_webserver, args=(plugins, port)
    )
    daemon.setDaemon(True)
    daemon.start()


def initialize(
    plugins: Optional[List[base_plugin.BasePlugin]] = list(ALL_PLUGINS.keys()),
    port: int = 6010
):
    """Initialize the cloud_training_tools SDK.

    The SDK will initialize the various diagnostic tools
    available for cloud training.

    Args:
      plugins: A list of plugins used in the cloud_training_tools tool. By default, this
        defaults to all plugins added to the `ALL_PLUGINS` variable.
      port: A port to serve web requests.
    """

    valid_plugins = []

    plugin_names = set()

    for plugin_name in plugins:
        plugin = ALL_PLUGINS.get(plugin_name)
        if not plugin:
            logging.warning(
                'Plugin %s does not exist. To add it, add it to the `ALL_PLUGINS` variable',
                plugin_name
            )
            continue

        if plugin.PLUGIN_NAME in plugin_names:
            logging.warning(
                "Plugin %s already exists, will not load", plugin.PLUGIN_NAME
            )
            continue

        # Checks for any libraries, versioning, etc. to use plugin
        if not plugin.can_initialize():
            logging.warning("Failed to initialize %s", plugin.PLUGIN_NAME)
            continue

        # Anything that must be run before the plugin runs
        plugin.setup()

        valid_plugins.append(plugin)
        plugin_names.add(plugin.PLUGIN_NAME)

    if not valid_plugins:
        logging.warning("No valid plugins, will not start cloud tools")
        return

    _run_app_thread(valid_plugins, port)
