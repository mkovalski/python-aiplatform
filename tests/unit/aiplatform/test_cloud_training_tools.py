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
from google.cloud.aiplatform.training_utils.cloud_training_tools.plugins.tf_profiler import tf_profiler
from google.cloud.aiplatform.training_utils.cloud_training_tools.plugins.tf_profiler.tf_profiler import (
    TFProfiler,
)
from google.cloud.aiplatform.training_utils.cloud_training_tools.plugins.tf_profiler import tensorboard_api
from google.cloud.aiplatform.training_utils.cloud_training_tools import web_server
from google.cloud.aiplatform.training_utils.cloud_training_tools import cloud_initializer


_ENV_VARS = training_utils.EnvironmentVariables()

_CLUSTER_SPEC_VM = '{"cluster":{"master":["localhost:1234"]},"environment":"cloud","task":{"type":"master","index":0}}'

_CLUSTER_SPEC_DISTRIB = '{"cluster":{"workerpool0":["host1:2222"],"workerpool1":["host2:2222"]},"environment":"cloud","task":{"type":"workerpool0","index":0},"job":"{\\"python_module\\":\\"\\",\\"package_uris\\":[],\\"job_args\\":[]}"}'


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
def tensorboard_api_mock():
    with mock.patch.object(
        tensorboard_api,
        #google.cloud.aiplatform.cloud_training_tools.plugins.tf_profiler.tensoboard_api,
        "make_profile_request_sender",
    ) as sender_mock:
        sender_mock.return_value = mock.Mock()
        yield sender_mock


@pytest.fixture
def setupEnvVars():
    os.environ["AIP_TF_PROFILER_PORT"] = "6009"
    os.environ["AIP_TENSORBOARD_LOG_DIR"] = "tmp/"
    os.environ["AIP_TENSORBOARD_API_URI"] = "test_api_uri"
    os.environ["AIP_TENSORBOARD_RESOURCE_NAME"] = "projects/123/region/us-central1/tensorboards/mytb"
    os.environ["CLUSTER_SPEC"] = _CLUSTER_SPEC_VM
    os.environ["CLOUD_ML_JOB_ID"] = "myjob"


def test_get_hostnames_vm():
    mock_cluster_spec = {"CLUSTER_SPEC": _CLUSTER_SPEC_VM,
                         "AIP_TF_PROFILER_PORT": "6009"}
    with mock.patch.dict(os.environ, mock_cluster_spec):
        hosts = tf_profiler._get_hostnames()
    assert hosts == 'grpc://localhost:6009'

def test_get_hostnames_cluster():
    mock_cluster_spec = {"CLUSTER_SPEC": _CLUSTER_SPEC_DISTRIB,
                         "AIP_TF_PROFILER_PORT": "6009"}
    with mock.patch.dict(os.environ, mock_cluster_spec):
        hosts = tf_profiler._get_hostnames()
    assert hosts == 'grpc://host1:6009,grpc://host2:6009'

class TestProfilerPlugin:
    # Initializion tests
    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeProfilerPortUnset(self):
        os.environ.pop("AIP_TF_PROFILER_PORT")
        assert not TFProfiler.can_initialize()

    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeTBLogDirUnset(self):
        os.environ.pop("AIP_TENSORBOARD_LOG_DIR")
        assert not TFProfiler.can_initialize()

    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeTBAPIuriUnset(self):
        os.environ.pop("AIP_TENSORBOARD_API_URI")
        assert not TFProfiler.can_initialize()

    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeTBResourceNameUnset(self):
        os.environ.pop("AIP_TENSORBOARD_RESOURCE_NAME")
        assert not TFProfiler.can_initialize()

    @pytest.mark.usefixtures("setupEnvVars")
    def testCanInitializeJobIdUnset(self):
        os.environ.pop("CLOUD_ML_JOB_ID")
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

    # Tests for plugin
    @pytest.mark.usefixtures("profile_plugin_mock")
    @pytest.mark.usefixtures("tensorboard_api_mock")
    @pytest.mark.usefixtures("setupEnvVars")
    def testCaptureProfile(self):
        profiler = TFProfiler()
        environ = dict(QUERY_STRING="?service_addr=myhost1,myhost2&someotherdata=5")
        start_response = None

        resp = profiler.capture_profile_wrapper(environ, start_response)
        assert resp[0].status_code == 200

    @pytest.mark.usefixtures("profile_plugin_mock")
    @pytest.mark.usefixtures("tensorboard_api_mock")
    @pytest.mark.usefixtures("setupEnvVars")
    def testCaptureProfileNoClusterSpec(self):
        profiler = TFProfiler()

        environ = dict(QUERY_STRING="?service_addr=myhost1,myhost2&someotherdata=5")
        start_response = None

        with mock.patch.dict(os.environ, {"CLUSTER_SPEC": "{}"}):
            resp = profiler.capture_profile_wrapper(environ, start_response)

        assert resp.status_code == 500

    @pytest.mark.usefixtures("profile_plugin_mock")
    @pytest.mark.usefixtures("tensorboard_api_mock")
    @pytest.mark.usefixtures("setupEnvVars")
    def testCaptureProfileNoCluster(self):

        profiler = TFProfiler()

        environ = dict(QUERY_STRING="?service_addr=myhost1,myhost2&someotherdata=5")
        start_response = None

        with mock.patch.dict(os.environ, {"CLUSTER_SPEC": '{"cluster": {}}'}):
            resp = profiler.capture_profile_wrapper(environ, start_response)

        assert resp.status_code == 500

    @pytest.mark.usefixtures("profile_plugin_mock")
    @pytest.mark.usefixtures("tensorboard_api_mock")
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


def test_start_cloud_training_tools_no_plugins():
    with mock.patch.object(
        cloud_initializer, "_run_app_thread", return_value=None
    ) as mock_app_thread:
        cloud_initializer.initialize(plugins=[])
        assert mock_app_thread.call_count == 0


def test_start_cloud_training_tools_bad_plugin():
    mock_plugin = mock.Mock()
    mock_plugin.can_initialize.return_value = True
    mock_plugin.setup.return_value = None

    mock_map = {"plugin1": mock_plugin}

    with mock.patch.object(
        cloud_initializer, "_run_app_thread", return_value=None
    ) as mock_app_thread:
        with mock.patch.dict(cloud_initializer.ALL_PLUGINS, mock_map):
            cloud_initializer.initialize(plugins=["plugin1", "plugin2"])
        mock_app_thread.assert_called_with([mock_plugin], 6010)


def test_start_cloud_training_tools_duplicate_plugins():

    mock_plugin = mock.Mock()
    mock_plugin.can_initialize.return_value = True
    mock_plugin.setup.return_value = None

    mock_map = {"plugin1": mock_plugin}

    with mock.patch.object(
        cloud_initializer, "_run_app_thread", return_value=None
    ) as mock_app_thread:
        with mock.patch.dict(cloud_initializer.ALL_PLUGINS, mock_map):
            cloud_initializer.initialize(plugins=["plugin1", "plugin1"])
        mock_app_thread.assert_called_with([mock_plugin], 6010)


def test_start_cloud_training_tools_fail_initiliaze_plugins():

    mock_plugin = mock.Mock()
    mock_plugin.can_initialize.return_value = True
    mock_plugin.setup.return_value = None

    mock_plugin_fail = mock.Mock()
    mock_plugin_fail.can_initialize.return_value = False
    mock_plugin_fail.setup.return_value = None

    mock_map = {"plugin1": mock_plugin, "plugin2": mock_plugin_fail}

    with mock.patch.object(
        cloud_initializer, "_run_app_thread", return_value=None
    ) as mock_app_thread:
        with mock.patch.dict(cloud_initializer.ALL_PLUGINS, mock_map):
            cloud_initializer.initialize(plugins=["plugin1", "plugin2"])
        mock_app_thread.assert_called_with([mock_plugin], 6010)


def test_run_app_thread():
    mock_item = mock.Mock()
    with mock.patch.object(threading, "Thread", return_value=mock_item):
        cloud_initializer._run_app_thread([], 6009)
        assert mock_item.start.call_count == 1


def test_run_webserver():
    mock_object = mock.Mock()
    with mock.patch.object(werkzeug.serving, "run_simple", mock_object):
        cloud_initializer._run_webserver([], 6009)
        assert mock_object.call_count == 1
