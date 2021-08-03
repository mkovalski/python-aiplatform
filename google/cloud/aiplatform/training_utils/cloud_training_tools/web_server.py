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

"""A basic webserver for hosting plugin routes."""

import os

from google.cloud.aiplatform.training_utils.cloud_training_tools.plugins import (
    base_plugin,
)
from typing import List
from werkzeug import wrappers


def create_web_server(plugins: List[base_plugin.BasePlugin]):
    """Create plugins and return a web app."""
    loaded_plugins = []

    for plugin in plugins:
        loaded_plugin = plugin()
        loaded_plugins.append(loaded_plugin)

    return WebServer(loaded_plugins)


class WebServer:
    """A basic web server for handling requests."""

    def __init__(self, plugins):
        """Creates a web server to host plugin routes.

        Args:
            plugins: A list of `plugins.BasePlugin`.

        Raises:
            ValueError: When there is an invalid route passed from
              one of the plugins.
        """

        self._plugins = plugins
        self._routes = {}

        # Routes are in form {plugin_name}/{route}
        for plugin in self._plugins:
            for route, handler in plugin.get_routes().items():
                if not route.startswith("/"):
                    raise ValueError(
                        'Routes should start with a "/", '
                        "invalid route for plugin %s, route %s"
                        % (plugin.PLUGIN_NAME, route)
                    )

                app_route = os.path.join("/", plugin.PLUGIN_NAME)

                app_route += route
                self._routes[app_route] = handler

    def dispatch_request(self, environ, start_response):
        """Handles the routing of requests.

        Args:
            envrion: A `werkzeug.Environ` object.
            start_response: A callable that indicates the start of the response.

        Returns:
            A `werkzeug.Response`
        """
        # Check for existince of route
        request = wrappers.Request(environ)

        if request.path in self._routes:
            return self._routes[request.path](environ, start_response)

        response = wrappers.Response("Not Found", status=404)
        return response(environ, start_response)

    def wsgi_app(self, environ, start_response):
        response = self.dispatch_request(environ, start_response)
        return response

    def __call__(self, environ, start_response):
        return self.wsgi_app(environ, start_response)
