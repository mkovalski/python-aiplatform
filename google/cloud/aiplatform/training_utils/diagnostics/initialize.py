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

from enum import Enum
import logging
import threading
from typing import Optional, List
from werkzeug import serving


from google.cloud.aiplatform.training_utils.diagnostics import web_server
from google.cloud.aiplatform.training_utils.diagnostics.plugins import tf_profiler


class Plugins(Enum):
    """List of plugins that are available for diagnostics SDK."""
    TF_PROFILER = 1

# To add a new plugin, add the plugin to the enum class and add a mapping to the
# MAP_TO_PLUGIN dictionary.

MAP_TO_PLUGIN = {Plugins.TF_PROFILER: tf_profiler.TFProfiler}
ALL_PLUGINS = list(MAP_TO_PLUGIN)

def _run_webserver(plugins, port):
    """Run the webserver that hosts the various diagnostics plugins."""
    app = web_server.create_web_server(plugins)
    serving.run_simple('127.0.0.1', port, app)

def _run_app_thread(plugins: List[Plugins],
                    port: int) -> None:
    """Run the diagnostics web server in a separate thread."""
    daemon = threading.Thread(name='diagnostics_server',
                              target=_run_webserver,
                              args = (plugins, port))
    daemon.setDaemon(True)
    daemon.start()

def start_diagnostics(plugins: Optional[List[Plugins]] = ALL_PLUGINS,
                   port: int = 6010):
    """Initialize the diagnostics SDK.

    The SDK will initialize the various diagnostic tools
    available for cloud training.

    Args:
      plugins: A list of plugins used in the diagnostics tool. By default, this
        defaults to all plugins added to the `MAP_TO_PLUGIN` variable.
      port: A port to serve web requests.
    """


    valid_plugins = []

    plugin_names = set()

    for plugin_name in plugins:
        plugin = MAP_TO_PLUGIN.get(plugin_name, None)
        if not plugin:
            logging.warning('Plugin %s does not exist',
                            plugin_name)
            continue

        if plugin.PLUGIN_NAME in plugin_names:
            logging.warning('Plugin %s already exists, will not load',
                            plugin.PLUGIN_NAME)
            continue

        if not plugin.can_initialize():
            logging.warning('Failed to initialize %s',
                            plugin_name)
            continue

        plugin.setup()

        valid_plugins.append(plugin)
        plugin_names.add(plugin.PLUGIN_NAME)

    if not valid_plugins:
        logging.warning("No valid plugins")
        return

    _run_app_thread(valid_plugins, port)
