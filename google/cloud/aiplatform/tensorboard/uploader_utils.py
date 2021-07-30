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

"""Helper functions for tensorboard uploader."""
import contextlib
import json
import re
import time
from typing import Callable, Dict
import uuid

from tensorboard.util import tb_logging

from google.api_core import exceptions
from google.cloud import storage
from google.cloud.aiplatform.compat.services import tensorboard_service_client_v1beta1
from google.cloud.aiplatform.compat.types import (
    tensorboard_run_v1beta1 as tensorboard_run,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_service_v1beta1 as tensorboard_service,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_time_series_v1beta1 as tensorboard_time_series,
)

TensorboardServiceClient = tensorboard_service_client_v1beta1.TensorboardServiceClient

logger = tb_logging.get_logger()

def get_source_bucket(logdir: str) -> storage.Bucket:
    m = re.match(r"gs:\/\/(.*?)(?=\/|$)", logdir)
    if not m:
        return None
    bucket = storage.Client().bucket(m[1])
    return bucket

class ExistingResourceNotFoundError(RuntimeError):
    """Resource could not be created or retrieved."""


class ExperimentNotFoundError(RuntimeError):
    pass


class RunResourceManager(object):
    """Manager object for the creation and storing of run resources."""

    def __init__(self, api, experiment_resource_name):
        """Create a run resource manager object.

        Args:
            api: a `TensorBoardWriterService` stub instance
            experiment_resource_name: Name of the experiment resource of the form
                projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}
        """
        self._api = api
        self._experiment_resource_name = experiment_resource_name

        self._run_to_run_resource: Dict[str, tensorboard_run.TensorboardRun] = {}

    @property
    def run_to_run_resource(self):
        """Mapping from run resource name to a `TensorboardRun`."""
        return self._run_to_run_resource

    def create_or_get_run_resource(
        self, run_name: str,
    ):
        """Creates a new Run Resource in current Tensorboard Experiment resource.

        Args:
          run_name: The display name of this run.
        """
        tb_run = tensorboard_run.TensorboardRun()
        tb_run.display_name = run_name
        try:
            tb_run = self._api.create_tensorboard_run(
                parent=self._experiment_resource_name,
                tensorboard_run=tb_run,
                tensorboard_run_id=str(uuid.uuid4()),
            )
        except exceptions.InvalidArgument as e:
            # If the run name already exists then retrieve it
            if "already exist" in e.message:
                runs_pages = self._api.list_tensorboard_runs(
                    parent=self._experiment_resource_name
                )
                for tb_run in runs_pages:
                    if tb_run.display_name == run_name:
                        break

                if tb_run.display_name != run_name:
                    raise ExistingResourceNotFoundError(
                        "Run with name %s already exists but is not resource list."
                        % run_name
                    )
            else:
                raise

        self._run_to_run_resource[run_name] = tb_run


class TimeSeriesResourceManager(object):
    """Helper class managing Time Series resources."""

    def __init__(self, run_resource_id: str, api: TensorboardServiceClient):
        """Constructor for TimeSeriesResourceManager.

        Args:
          run_resource_id: The resource id for the run with the following format
            projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}
          api: TensorboardServiceStub
        """
        self._run_resource_id = run_resource_id
        self._api = api
        self._tag_to_time_series_proto: Dict[
            str, tensorboard_time_series.TensorboardTimeSeries
        ] = {}

    def get_or_create(
        self,
        tag_name: str,
        time_series_resource_creator: Callable[
            [], tensorboard_time_series.TensorboardTimeSeries
        ],
    ) -> tensorboard_time_series.TensorboardTimeSeries:
        """get a time series resource with given tag_name, and create a new one on

        OnePlatform if not present.

        Args:
          tag_name: The tag name of the time series in the Tensorboard log dir.
          time_series_resource_creator: A callable that produces a TimeSeries for
            creation.
        """
        if tag_name in self._tag_to_time_series_proto:
            return self._tag_to_time_series_proto[tag_name]

        time_series = time_series_resource_creator()
        time_series.display_name = tag_name
        try:
            time_series = self._api.create_tensorboard_time_series(
                parent=self._run_resource_id, tensorboard_time_series=time_series
            )
        except exceptions.InvalidArgument as e:
            # If the time series display name already exists then retrieve it
            if "already exist" in e.message:
                list_of_time_series = self._api.list_tensorboard_time_series(
                    request=tensorboard_service.ListTensorboardTimeSeriesRequest(
                        parent=self._run_resource_id,
                        filter="display_name = {}".format(json.dumps(str(tag_name))),
                    )
                )
                num = 0
                for ts in list_of_time_series:
                    time_series = ts
                    num += 1
                    break
                if num != 1:
                    raise ValueError(
                        "More than one time series resource found with display_name: {}".format(
                            tag_name
                        )
                    )
            else:
                raise

        self._tag_to_time_series_proto[tag_name] = time_series
        return time_series


@contextlib.contextmanager
def request_logger(request: tensorboard_service.WriteTensorboardRunDataRequest):
    """Context manager to log request size and duration."""
    upload_start_time = time.time()
    request_bytes = request._pb.ByteSize()  # pylint: disable=protected-access
    logger.info("Trying request of %d bytes", request_bytes)
    yield
    upload_duration_secs = time.time() - upload_start_time
    logger.info(
        "Upload of (%d bytes) took %.3f seconds", request_bytes, upload_duration_secs,
    )
