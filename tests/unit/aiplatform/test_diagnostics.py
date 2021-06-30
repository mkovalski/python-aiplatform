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

import importlib.util
import json
import os
import threading

import pytest
import unittest
import werkzeug

from unittest import mock
from werkzeug import wrappers
from werkzeug.test import EnvironBuilder

from google.cloud.aiplatform import training_utils
from google.cloud.aiplatform.training_utils.diagnostics.plugins.tf_profiler import (
    TFProfiler,
)
from google.cloud.aiplatform.training_utils.diagnostics import web_server
from google.cloud.aiplatform.training_utils.diagnostics import initialize


_ENV_VARS = training_utils.EnvironmentVariables()

_CLUSTER_SPEC = '{"cluster":{"master":["localhost:1234"],"worker":["localhost:3456"]},"environment":"cloud","task":{"type":"master","index":0}}'


@pytest.fixture
def profile_plugin_mock():
    import tensorboard_plugin_profile.profile_plugin

    with mock.patch.object(
        tensorboard_plugin_profile.profile_plugin.ProfilePlugin, "capture_route"
    ) as profile_mock:
        profile_mock.return_value = (
            wrappers.BaseResponse(
                json.dumps({"error": "some error"}),
                content_type="application/json",
                status=200,
            ),
        )
        yield profile_mock


@pytest.fixture
def setupEnvVars():
    os.environ["AIP_TF_PROFILER_PORT"] = "6009"
    os.environ["AIP_TENSORBOARD_LOG_DIR"] = "tmp/"
    os.environ["CLUSTER_SPEC"] = _CLUSTER_SPEC


class TestProfilerPlugin:
    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeProfilerPortUnset(self):
        os.environ.pop("AIP_TF_PROFILER_PORT")
        assert not TFProfiler.can_initialize()

    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeTBLogDirUnset(self):
        os.environ.pop("AIP_TENSORBOARD_LOG_DIR")
        assert not TFProfiler.can_initialize()

    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeTFInstalled(self):
        orig_find_spec = importlib.util.find_spec

        def tf_import_mock(name, *args, **kwargs):
            if name == "tensorflow":
                return None
            return orig_find_spec(name, *args, **kwargs)

        with mock.patch("importlib.util.find_spec", side_effect=tf_import_mock):
            assert not TFProfiler.can_initialize()

    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeTFVersion(self):
        import tensorflow

        with mock.patch.dict(tensorflow.__dict__, {"__version__": "1.2.3.4"}):
            assert not TFProfiler.can_initialize()

    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeOldTFVersion(self):
        import tensorflow

        with mock.patch.dict(tensorflow.__dict__, {"__version__": "1.13.0"}):
            assert not TFProfiler.can_initialize()

    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeUnknownTFVersion(self):
        import tensorflow

        with mock.patch.dict(tensorflow.__dict__, {"__version__": "0.13.0"}):
            assert not TFProfiler.can_initialize()

    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeNoClusterSpec(self):
        os.environ["CLUSTER_SPEC"] = "{}"
        assert not TFProfiler.can_initialize()

    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeNoProfilePlugin(self):
        orig_find_spec = importlib.util.find_spec

        def plugin_import_mock(name, *args, **kwargs):
            if name == "tensorboard_plugin_profile":
                return None
            return orig_find_spec(name, *args, **kwargs)

        with mock.patch("importlib.util.find_spec", side_effect=plugin_import_mock):
            assert not TFProfiler.can_initialize()

    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitialize(self):
        assert TFProfiler.can_initialize()

    def testSetup(self):
        import tensorflow

        with mock.patch.object(
            tensorflow.profiler.experimental.server, "start", return_value=None
        ) as server_mock:
            TFProfiler.setup()

            assert server_mock.call_count == 1

    @pytest.mark.usefixtures("profile_plugin_mock")
    @pytest.mark.usefixtures("setupEnvVars")
    def testCaptureProfile(self):
        profiler = TFProfiler()
        environ = dict(QUERY_STRING="?service_addr=myhost1,myhost2&someotherdata=5")
        start_response = None

        resp = profiler.capture_profile_wrapper(environ, start_response)
        assert resp[0].status_code == 200

    @pytest.mark.usefixtures("profile_plugin_mock")
    @pytest.mark.usefixtures("setupEnvVars")
    def testCaptureProfileNoClusterSpec(self):
        profiler = TFProfiler()

        environ = dict(QUERY_STRING="?service_addr=myhost1,myhost2&someotherdata=5")
        start_response = None

        with mock.patch.dict(os.environ, {"CLUSTER_SPEC": "{}"}):
            resp = profiler.capture_profile_wrapper(environ, start_response)

        assert resp.status_code == 500

    @pytest.mark.usefixtures("profile_plugin_mock")
    @pytest.mark.usefixtures("setupEnvVars")
    def testCaptureProfileNoCluster(self):

        profiler = TFProfiler()

        environ = dict(QUERY_STRING="?service_addr=myhost1,myhost2&someotherdata=5")
        start_response = None

        with mock.patch.dict(os.environ, {"CLUSTER_SPEC": '{"cluster": {}}'}):
            resp = profiler.capture_profile_wrapper(environ, start_response)

        assert resp.status_code == 500

    @pytest.mark.usefixtures("profile_plugin_mock")
    @pytest.mark.usefixtures("setupEnvVars")
    def testCaptureProfileNoMaster(self):
        profiler = TFProfiler()

        environ = dict(QUERY_STRING="?service_addr=myhost1,myhost2&someotherdata=5")
        start_response = None

        with mock.patch.dict(
            os.environ, {"CLUSTER_SPEC": '{"cluster": {"foo": "bar"}}'}
        ):
            resp = profiler.capture_profile_wrapper(environ, start_response)

        assert resp.status_code == 500

    @pytest.mark.usefixtures("profile_plugin_mock")
    @pytest.mark.usefixtures("setupEnvVars")
    def testGetRoutes(self):
        profiler = TFProfiler()

        routes = profiler.get_routes()
        assert isinstance(routes, dict)


class TestWebServer(unittest.TestCase):
    def test_create_webserver_bad_route(self):
        plugin = mock.Mock()
        plugin.get_routes.return_value = {"my_route": "some_handler"}

        self.assertRaises(ValueError, web_server.WebServer, [plugin])

    def test_dispatch_bad_request(self):
        ws = web_server.create_web_server([])

        builder = EnvironBuilder(method="GET", path="/")

        env = builder.get_environ()

        # Mock a start response callable
        response = []
        buff = []

        def start_response(status, headers):
            response[:] = [status, headers]
            return buff.append

        ws(env, start_response)

        assert response[0] == "404 NOT FOUND"

    def test_correct_response(self):
        res_dict = {"response": "OK"}

        def my_callable(var1, var2):
            return wrappers.BaseResponse(
                json.dumps(res_dict), content_type="application/json", status=200
            )

        def my_plugin():
            plugin = mock.Mock()
            plugin.get_routes.return_value = {"/my_route": my_callable}
            plugin.PLUGIN_NAME = "my_plugin"
            return plugin

        ws = web_server.create_web_server([my_plugin])

        builder = EnvironBuilder(method="GET", path="/my_plugin/my_route")

        env = builder.get_environ()

        # Mock a start response callable
        response = []
        buff = []

        def start_response(status, headers):
            response[:] = [status, headers]
            return buff.append

        res = ws(env, start_response)

        final_response = json.loads(res.response[0].decode("utf-8"))

        assert final_response == res_dict


def test_start_diagnostics_no_plugins():
    with mock.patch.object(
        initialize, "_run_app_thread", return_value=None
    ) as mock_app_thread:
        initialize.start_diagnostics(plugins=[])
        assert mock_app_thread.call_count == 0


def test_start_diagnostics_bad_plugin():
    mock_plugin = mock.Mock()
    mock_plugin.can_initialize.return_value = True
    mock_plugin.setup.return_value = None

    mock_map = {"plugin1": mock_plugin}

    with mock.patch.object(
        initialize, "_run_app_thread", return_value=None
    ) as mock_app_thread:
        with mock.patch.dict(initialize.MAP_TO_PLUGIN, mock_map):
            initialize.start_diagnostics(plugins=["plugin1", "plugin2"])
        mock_app_thread.assert_called_with([mock_plugin], 6010)


def test_start_diagnostics_duplicate_plugins():

    mock_plugin = mock.Mock()
    mock_plugin.can_initialize.return_value = True
    mock_plugin.setup.return_value = None

    mock_map = {"plugin1": mock_plugin}

    with mock.patch.object(
        initialize, "_run_app_thread", return_value=None
    ) as mock_app_thread:
        with mock.patch.dict(initialize.MAP_TO_PLUGIN, mock_map):
            initialize.start_diagnostics(plugins=["plugin1", "plugin1"])
        mock_app_thread.assert_called_with([mock_plugin], 6010)


def test_start_diagnostics_fail_initiliaze_plugins():

    mock_plugin = mock.Mock()
    mock_plugin.can_initialize.return_value = True
    mock_plugin.setup.return_value = None

    mock_plugin_fail = mock.Mock()
    mock_plugin_fail.can_initialize.return_value = False
    mock_plugin_fail.setup.return_value = None

    mock_map = {"plugin1": mock_plugin, "plugin2": mock_plugin_fail}

    with mock.patch.object(
        initialize, "_run_app_thread", return_value=None
    ) as mock_app_thread:
        with mock.patch.dict(initialize.MAP_TO_PLUGIN, mock_map):
            initialize.start_diagnostics(plugins=["plugin1", "plugin2"])
        mock_app_thread.assert_called_with([mock_plugin], 6010)


def test_run_app_thread():
    mock_item = mock.Mock()
    with mock.patch.object(threading, "Thread", return_value=mock_item):
        initialize._run_app_thread([], 6009)
        assert mock_item.start.call_count == 1


def test_run_webserver():
    mock_object = mock.Mock()
    with mock.patch.object(werkzeug.serving, "run_simple", mock_object):
        initialize._run_webserver([], 6009)
        assert mock_object.call_count == 1
