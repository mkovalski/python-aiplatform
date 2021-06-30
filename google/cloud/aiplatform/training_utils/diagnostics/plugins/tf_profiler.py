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

"""A plugin to handle remote tensoflow profiler sessions for Vertex AI."""

import argparse
from collections import namedtuple
import json
import logging
import os
from tensorboard.plugins.base_plugin import TBContext
from typing import Dict, Optional
from urllib import parse
import importlib.util
from werkzeug import wrappers

from google.cloud.aiplatform.training_utils import EnvironmentVariables
from google.cloud.aiplatform.training_utils.diagnostics.plugins import base_plugin


# Simple namedtuple for tf verison information.
Version = namedtuple("Version", ["major", "minor", "patch"])

# Environment variables from training jobs.
_ENV_VARS = EnvironmentVariables()

logger = logging.Logger("tf-profiler")


def _tf_installed() -> bool:
    """Helper function to determine if tensorflow is installed."""
    if not importlib.util.find_spec('tensorflow'):
        logger.warning("Could not import tensorflow, will not run tf profiling service")
        return False
    return True


def _get_tf_versioning() -> Optional[Version]:
    """Convert version string to a Version namedtuple for ease of parsing."""
    import tensorflow as tf

    version = tf.__version__

    versioning = version.split(".")
    try:
        assert len(versioning) == 3
    except AssertionError:
        logger.warning(
            "Could not find major, minor, and patch versions of tensorflow. Version found: %s",
            version,
        )
        return

    return Version(int(versioning[0]), int(versioning[1]), int(versioning[2]))


def _is_compatible_version(version: Version) -> bool:
    """Check if version is compatible with tf profiling.

    Profiling plugin is available to be used for version >= 2.2.0.

    Args:
        version: `Verison` of tensorflow.

    Returns:
        If compatible with profiler.
    """
    if version.major >= 2 and version.minor >= 2:
        return True
    logger.warning("Tensorflow version is not compatible with TF profiler")
    return False


def _create_profiling_context() -> TBContext:
    """Creates the base context needed for TB Profiler."""

    context_flags = argparse.Namespace(master_tpu_unsecure_channel=None)

    context = TBContext(
        logdir=_ENV_VARS.tensorboard_log_dir, multiplexer=None, flags=context_flags
    )

    return context


def _check_cluster_spec() -> Optional[str]:
    cluster_spec = json.loads(os.environ.get("CLUSTER_SPEC", "{}"))
    if not cluster_spec:
        return 'Environment variable "CLUSTER_SPEC" is not set'


def _host_to_grpc(hostname: str) -> str:
    """Format a hostname to a grpc address.

    Args:
        hostname: address as a string

    Returns:
        address in form of: 'grpc://{hostname}:{port}'
    """
    return (
        "grpc://" + "".join(hostname.split(":")[:-1]) + ":" + _ENV_VARS.tf_profiler_port
    )


def _get_master_host() -> Optional[str]:
    """Get the master service address from an environment variable.

    Currently, only profile the master host.

    Returns:
        A master host formatted by `_host_to_grpc`.
    """
    cluster_spec = _get_cluster_spec()
    if not cluster_spec:
        return

    cluster = cluster_spec.get("cluster", "")
    if not cluster:
        return

    host_list = cluster.get("master", [])
    if not host_list:
        return

    return _host_to_grpc(host_list[0])


def _get_cluster_spec() -> Optional[Dict[str, str]]:
    """Get the cluster spec so we can profile multiple workers."""
    cluster_spec = json.loads(os.environ.get("CLUSTER_SPEC", "{}"))
    return cluster_spec


def _update_environ(environ) -> str:
    """Add parameters to the query that are retrieved from training side."""
    host = _get_master_host()

    if not host:
        return "Could not get the master host"

    query_dict = {}
    query_dict["service_addr"] = host

    # Update service address and worker list
    # Use parse_qsl and then convert list to dictionary so we can update
    # attributes
    prev_query_string = dict(parse.parse_qsl(environ["QUERY_STRING"]))
    prev_query_string.update(query_dict)

    environ["QUERY_STRING"] = parse.urlencode(prev_query_string)

    return ""


class TFProfiler(base_plugin.BasePlugin):
    """Handler for Tensorflow Profiling."""

    PLUGIN_NAME = "profile"

    def __init__(self):
        """Build a TFProfiler object."""
        from tensorboard_plugin_profile.profile_plugin import ProfilePlugin

        context = _create_profiling_context()
        self.profile_plugin = ProfilePlugin(context)

    def get_routes(self):
        """List of routes to serve."""
        return {"/capture_profile": self.capture_profile_wrapper}

    # Define routes below
    def capture_profile_wrapper(self, environ, start_response):
        """Take a request from tensorboard.gcp and run the profiling for the available servers."""
        # The service address (localhost) and worker list are populated locally
        update_environ_error = _update_environ(environ)

        if update_environ_error:
            err = {"error": "Could not parse the environ: %s" % update_environ_error}
            return wrappers.BaseResponse(
                json.dumps(err), content_type="application/json", status=500
            )

        response = self.profile_plugin.capture_route(environ, start_response)

        return response

    # End routes

    @staticmethod
    def setup() -> None:
        import tensorflow as tf

        tf.profiler.experimental.server.start(int(_ENV_VARS.tf_profiler_port))

    @staticmethod
    def can_initialize() -> bool:
        """Check that we can use the TF Profiler plugin.

        This function checks a number of dependencies for the plugin to ensure we have the
        right packages installed, the necessary versions, and the correct environment variables set.

            - `EnvironmentVariables().tf_profiler_port` must be set, set by
              Vertex AI.
            - `EnvironmentVariables().tensorboard_log_dir` must be set, set by
              Vertex AI when run with tensorboard option.
            - `tensorboard_plugin_profile` must be installed.
            - Tensorflow >= 2.2.0
            - 'CLUSTER_SPEC' environment variable must be set, set by Vertex AI.
        """

        # Environment variable checks
        # Check that AI Platform service set a port for TF profiling
        if _ENV_VARS.tf_profiler_port is None:
            logger.warning(
                '"%s" environment variable not set, cannot enable profiling.',
                "AIP_TF_PROFILER_PORT",
            )
            return False

        # Check that a log directory was specified
        if _ENV_VARS.tensorboard_log_dir is None:
            logger.warning(
                "Must set a tensorboard log directory, "
                "run training with tensorboard enabled."
            )
            return False

        # Check tf is installed
        if not _tf_installed():
            return False

        # Check tensorflow version, introduced 1.14 >=
        version = _get_tf_versioning()
        if not version:
            return False

        if not _is_compatible_version(version):
            logger.warning(
                "Version %s is incompatible with tf profiler."
                "To use the profiler, choose a version >= 2.2.0",
                version,
            )
            return False

        # Check to make sure CLUSTER_SPEC is set
        # Details on CLUSTER_SPEC: https://cloud.google.com/ai-platform/training/docs/distributed-training-containers#about-cluster-spec
        cluster_spec_error = _check_cluster_spec()
        if cluster_spec_error:
            logger.warning(cluster_spec_error)
            return False

        # Check for the tf profiler plugin
        if not importlib.util.find_spec('tensorboard_plugin_profile'):
            logger.warning(
                "Could not import tensorboard_plugin_profile, will not run tf profiling service"
            )
            return False

        return True
