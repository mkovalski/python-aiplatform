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
from typing import Dict, Optional, Tuple
from urllib import parse
from werkzeug import wrappers

from google.cloud.aiplatform.training_utils import EnvironmentVariables
from google.cloud.aiplatform.training_utils.debugging.plugins import base_plugin


# Simple namedtuple for tf verison information.
Version = namedtuple('Version', ['major', 'minor', 'patch'])

# Environment variables from training jobs.
_ENV_VARS = EnvironmentVariables()

def _tf_installed() -> bool:
    """Helper function to determine if tensorflow is installed."""
    try:
        import tensorflow as tf
    except ImportError as e:
        logging.warning('Could not import tensorflow, will not run tf profiling service')
        return False
    return True

def _get_tf_versioning() -> Optional[Version]:
    """Convert version string to a Version namedtuple for ease of parsing."""
    import tensorflow as tf
    version = tf.__version__

    versioning = version.split('.')
    try:
        assert(len(versioning) == 3)
    except AssertionError as error:
        logging.warning(
            'Could not find major, minor, and patch versions of tensorflow. Version found: %s',
            version)
        return

    return Version(int(versioning[0]),
                   int(versioning[1]),
                   int(versioning[2]))

def _is_compatible_version(version: Version) -> bool:
    """Check if version is compatible with tf profiling.

    Profiling plugin is available to be used for version >= 2.2.0.
    """
    if version.major >= 2 and version.minor >= 2:
        return True
    return False

def _create_debugging_context() -> TBContext:
    """Creates the base context needed for TB Profiler."""

    context_flags = argparse.Namespace(
        master_tpu_unsecure_channel=None)

    context = TBContext(logdir=None,
                        multiplexer=None,
                        flags=context_flags)

    return context

def _check_cluster_spec() -> Optional[str]:
    cluster_spec = json.loads(os.environ.get('CLUSTER_SPEC', '{}'))
    if not cluster_spec:
        return 'Environment variable "CLUSTER_SPEC" is not set'

def _host_to_grpc(hostname: str) -> str:
    return 'grpc://' + ''.join(hostname.split(':')[:-1]) + ':' + _ENV_VARS.tf_profiler_port

def _get_hosts_list() -> Optional[str]:
    """Get the service address from an environment variable."""
    cluster_spec = _get_cluster_spec()
    if not cluster_spec:
        return

    cluster = cluster_spec.get('cluster', '')
    if not cluster:
        return

    host_list = cluster.get('master', [])
    if not host_list:
        return

    # If no workers, this is OK. Dealing with single host.
    worker_list = cluster.get('worker', [])

    host_list.extend(worker_list)

    for i in range(len(host_list)):
        host_list[i] = _host_to_grpc(host_list[i])

    return host_list

def _get_cluster_spec() -> Optional[Dict[str, str]]:
    '''Get the cluster spec so we can profile multiple workers.'''
    cluster_spec = json.loads(os.environ.get('CLUSTER_SPEC', '{}'))
    return cluster_spec

def _update_environ(environ) -> str:
    """Add parameters to the query that are retrieved from training side."""
    hosts = _get_hosts_list()

    if not hosts:
      return 'Could not get the hosts list'

    query_dict = {}
    query_dict['service_addr'] = ','.join(hosts)

    # Update service address and worker list
    # Use parse_qsl and then convert list to dictionary so we can update
    # attributes
    prev_query_string = dict(parse.parse_qsl(environ['QUERY_STRING']))
    prev_query_string.update(query_dict)

    environ['QUERY_STRING'] = parse.urlencode(prev_query_string)

    return ''

class TFProfiler(base_plugin.BasePlugin):
    """Handler for Tensorflow Profiling."""
    PLUGIN_NAME = 'profile'

    def __init__(self):
        from tensorboard_plugin_profile.profile_plugin import ProfilePlugin
        context = _create_debugging_context()
        self.profile_plugin = ProfilePlugin(context)

    def get_routes(self):
        """List of routes to serve."""
        return {'/capture_profile' : self.capture_profile_wrapper}

    # Define routes below
    def capture_profile_wrapper(self, environ, start_response):
        """Take a request from tensorboard.gcp and run the profiling for the available servers."""
        # The service address (localhost) and worker list are populated locally
        update_environ_error = _update_environ(environ)

        if update_environ_error:
            err = {'error': 'Could not parse the environ: %s' % update_environ_error}
            return wrappers.BaseResponse(json.dumps(err),
                                         content_type = 'application/json',
                                         status = 500)

        # Create a temporary directory to store the profiler logs. This is done so
        # that any logs created will be fully written out before the tensorboard
        # log uploader picks them up.
        self.profile_plugin.logdir = _ENV_VARS.tensorboard_log_dir
        response = self.profile_plugin.capture_route(environ, start_response)
        return response

    # End routes

    @staticmethod
    def setup() -> None:
        import tensorflow as tf
        tf.profiler.experimental.server.start(
            int(_ENV_VARS.tf_profiler_port))

    @staticmethod
    def can_initialize() -> bool:
        """Check that we can use the TF Profiler plugin.

        This function checks a number of dependencies for the plugin to ensure we have the
        right packages installed, the necessary versions, and the correct environment variables set.
        """

        # Environment variable checks
        # Check that AI Platform service set a port for TF profiling
        if _ENV_VARS.tf_profiler_port is None:
            logging.warning('"%s" environment variable not set, cannot enable profiling.',
                           _ENV_VARS.tf_profiler_port,
                           RuntimeWarning)
            return False

        # Check that a log directory was specified
        if _ENV_VARS.tensorboard_log_dir is None:
            logging.warning('Must set a tensorboard log directory')
            return False

        # Check tf is installed
        if not _tf_installed():
            return False

        # Check tensorflow version, introduced 1.14 >=
        version = _get_tf_versioning()
        if not version:
            return False

        if not _is_compatible_version(version):
            logging.warning('Version %s is incompatible with tf profiler.'
                            'To use the profiler, choose a version >= 2.2.0',
                            version)
            return False

        # Check to make sure CLUSTER_SPEC is set
        # Details on CLUSTER_SPEC: https://cloud.google.com/ai-platform/training/docs/distributed-training-containers#about-cluster-spec

        cluster_spec_error = _check_cluster_spec()
        if cluster_spec_error:
            warnings.warn(cluster_spec_error)
            return False

        # Check for the tf profiler plugin
        try:
            from tensorboard_plugin_profile.profile_plugin import ProfilePlugin
        except ImportError as e:
          logging.warning("Could not import tensorboard_plugin_profile, will not run tf profiling service",
                          RuntimeWarning)
          return False

        return True

