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
"""Base request sender for different tensorboard events."""
import abc
from typing import ContextManager

from tensorboard.uploader import util
from tensorboard.uploader import upload_tracker
import tensorflow as tf

from google.cloud.aiplatform.compat.services import tensorboard_service_client_v1beta1
from google.cloud.aiplatform.compat.types import (
    tensorboard_data_v1beta1 as tensorboard_data,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_service_v1beta1 as tensorboard_service,
)
from google.cloud.aiplatform.compat.types import (
    tensorboard_time_series_v1beta1 as tensorboard_time_series,
)

TensorboardServiceClient = tensorboard_service_client_v1beta1.TensorboardServiceClient

class Sender(object):
    """Base sender class for typing purposes."""
    pass


class BaseBatchedRequestSender(object):
    """Helper class for building requests that fit under a size limit.

    This class accumulates a current request.  `add_event(...)` may or may not
    send the request (and start a new one).  After all `add_event(...)` calls
    are complete, a final call to `flush()` is needed to send the final request.

    This class is not threadsafe. Use external synchronization if calling its
    methods concurrently.
    """

    def __init__(
        self,
        run_resource_id: str,
        api: TensorboardServiceClient,
        rpc_rate_limiter: util.RateLimiter,
        max_request_size: int,
        tracker: upload_tracker.UploadTracker,
    ):
        """Constructor for _BaseBatchedRequestSender.

        Args:
          run_resource_id: The resource id for the run with the following format
            projects/{project}/locations/{location}/tensorboards/{tensorboard}/experiments/{experiment}/runs/{run}
          api: TensorboardServiceStub
          rpc_rate_limiter: until.RateLimiter to limit rate of this request sender
          max_request_size: max number of bytes to send
          tracker:
        """
        self._run_resource_id = run_resource_id
        self._api = api
        self._rpc_rate_limiter = rpc_rate_limiter
        self._byte_budget_manager = _ByteBudgetManager(max_request_size)
        self._tracker = tracker

        # cache: map from Tensorboard tag to TimeSeriesData
        # cleared whenever a new request is created
        self._tag_to_time_series_data: Dict[str, tensorboard_data.TimeSeriesData] = {}

        self._time_series_resource_manager = uploader_utils.TimeSeriesResourceManager(
            self._run_resource_id, self._api
        )
        self._new_request()

    def _new_request(self):
        """Allocates a new request and refreshes the budget."""
        self._request = tensorboard_service.WriteTensorboardRunDataRequest()
        self._tag_to_time_series_data.clear()
        self._num_values = 0
        self._request.tensorboard_run = self._run_resource_id

    def add_event(
        self,
        event: tf.compat.v1.Event,
        value: tf.compat.v1.Summary.Value,
        metadata: tf.compat.v1.SummaryMetadata,
    ):
        """Attempts to add the given event to the current request.
        If the event cannot be added to the current request because the byte
        budget is exhausted, the request is flushed, and the event is added
        to the next request.
        Args:
          event: The tf.compat.v1.Event event containing the value.
          value: A scalar tf.compat.v1.Summary.Value.
          metadata: SummaryMetadata of the event.
        """
        try:
            self._add_event_internal(event, value, metadata)
        except _OutOfSpaceError:
            self.flush()
            # Try again.  This attempt should never produce OutOfSpaceError
            # because we just flushed.
            try:
                self._add_event_internal(event, value, metadata)
            except _OutOfSpaceError:
                raise RuntimeError("add_event failed despite flush")

    def _add_event_internal(
        self,
        event: tf.compat.v1.Event,
        value: tf.compat.v1.Summary.Value,
        metadata: tf.compat.v1.SummaryMetadata,
    ):
        self._num_values += 1
        time_series_data_proto = self._tag_to_time_series_data.get(value.tag)
        if time_series_data_proto is None:
            time_series_data_proto = self._create_time_series_data(value.tag, metadata)
        self._create_point(time_series_data_proto, event, value, metadata)

    def flush(self):
        """Sends the active request after removing empty runs and tags.
        Starts a new, empty active request.
        """
        request = self._request
        request.time_series_data = list(self._tag_to_time_series_data.values())
        _prune_empty_time_series(request)
        if not request.time_series_data:
            return

        self._rpc_rate_limiter.tick()

        with _request_logger(request):
            with self._get_tracker():
                try:
                    self._api.write_tensorboard_run_data(
                        tensorboard_run=self._run_resource_id,
                        time_series_data=request.time_series_data,
                    )
                except grpc.RpcError as e:
                    if (
                        hasattr(e, "code")
                        and getattr(e, "code")() == grpc.StatusCode.NOT_FOUND
                    ):
                        raise ExperimentNotFoundError()
                    logger.error("Upload call failed with error %s", e)

        self._new_request()

    def _create_time_series_data(
        self, tag_name: str, metadata: tf.compat.v1.SummaryMetadata
    ) -> tensorboard_data.TimeSeriesData:
        """Adds a time_series for the tag_name, if there's space.
        Args:
          tag_name: String name of the tag to add (as `value.tag`).
        Returns:
          The TimeSeriesData in _request proto with the given tag name.
        Raises:
          _OutOfSpaceError: If adding the tag would exceed the remaining
            request budget.
        """
        time_series_data_proto = tensorboard_data.TimeSeriesData(
            tensorboard_time_series_id=self._time_series_resource_manager.get_or_create(
                tag_name,
                lambda: tensorboard_time_series.TensorboardTimeSeries(
                    display_name=tag_name,
                    value_type=self._value_type,
                    plugin_name=metadata.plugin_data.plugin_name,
                    plugin_data=metadata.plugin_data.content,
                ),
            ).name.split("/")[-1],
            value_type=self._value_type,
        )

        self._byte_budget_manager.add_time_series(time_series_data_proto)
        self._tag_to_time_series_data[tag_name] = time_series_data_proto
        return time_series_data_proto

    def _create_point(
        self,
        time_series_proto: tensorboard_data.TimeSeriesData,
        event: tf.compat.v1.Event,
        value: tf.compat.v1.Summary.Value,
        metadata: tf.compat.v1.SummaryMetadata,
    ):
        """Adds a scalar point to the given tag, if there's space.
        Args:
          time_series_proto: TimeSeriesData proto to which to add a point.
          event: Enclosing `Event` proto with the step and wall time data.
          value: `Summary.Value` proto.
          metadata: SummaryMetadata of the event.
        Raises:
          _OutOfSpaceError: If adding the point would exceed the remaining
            request budget.
        """
        point = self._create_data_point(event, value, metadata)

        if not self._validate(point, event, value):
            return

        time_series_proto.values.extend([point])
        try:
            self._byte_budget_manager.add_point(point)
        except _OutOfSpaceError:
            time_series_proto.values.pop()
            raise

    @abc.abstractmethod
    def _get_tracker(self) -> ContextManager:
        """
        :return: tracker function from upload_tracker.UploadTracker
        """
        pass

    @property
    @classmethod
    @abc.abstractmethod
    def _value_type(cls,) -> tensorboard_time_series.TensorboardTimeSeries.ValueType:
        """
        :return: Value type of the time series.
        """
        pass

    @abc.abstractmethod
    def _create_data_point(
        self,
        event: tf.compat.v1.Event,
        value: tf.compat.v1.Summary.Value,
        metadata: tf.compat.v1.SummaryMetadata,
    ) -> tensorboard_data.TimeSeriesDataPoint:
        """
        Creates data point protos for sending to the OnePlatform API.
        """
        pass

    def _validate(
        self,
        point: tensorboard_data.TimeSeriesDataPoint,
        event: tf.compat.v1.Event,
        value: tf.compat.v1.Summary.Value,
    ):
        """
        Validations performed before including the data point to be sent to the
        OnePlatform API.
        """
        return True


class _ByteBudgetManager(object):
    """Helper class for managing the request byte budget for certain RPCs.

    This should be used for RPCs that organize data by Runs, Tags, and Points,
    specifically WriteScalar and WriteTensor.

    Any call to add_time_series() or add_point() may raise an
    _OutOfSpaceError, which is non-fatal. It signals to the caller that they
    should flush the current request and begin a new one.

    For more information on the protocol buffer encoding and how byte cost
    can be calculated, visit:

    https://developers.google.com/protocol-buffers/docs/encoding
    """

    def __init__(self, max_bytes: int):
        # The remaining number of bytes that we may yet add to the request.
        self._byte_budget = None  # type: int
        self._max_bytes = max_bytes

    def reset(self, base_request: tensorboard_service.WriteTensorboardRunDataRequest):
        """Resets the byte budget and calculates the cost of the base request.

        Args:
          base_request: Base request.

        Raises:
          _OutOfSpaceError: If the size of the request exceeds the entire
            request byte budget.
        """
        self._byte_budget = self._max_bytes
        self._byte_budget -= (
            base_request._pb.ByteSize()
        )  # pylint: disable=protected-access
        if self._byte_budget < 0:
            raise _OutOfSpaceError("Byte budget too small for base request")

    def add_time_series(self, time_series_proto: tensorboard_data.TimeSeriesData):
        """Integrates the cost of a tag proto into the byte budget.

        Args:
          time_series_proto: The proto representing a time series.

        Raises:
          _OutOfSpaceError: If adding the time_series would exceed the remaining
          request budget.
        """

    def add_point(self, point_proto: tensorboard_data.TimeSeriesDataPoint):
        """Integrates the cost of a point proto into the byte budget.
        Args:
          point_proto: The proto representing a point.
        Raises:
          _OutOfSpaceError: If adding the point would exceed the remaining request
           budget.
        """
        submessage_cost = point_proto._pb.ByteSize()  # pylint: disable=protected-access
        cost = (
            # The size of the point proto.
            submessage_cost
            # The size of the varint that describes the length of the point
            # proto.
            + _varint_cost(submessage_cost)
            # The size of the proto key.
            + 1
        )
        if cost > self._byte_budget:
            raise _OutOfSpaceError()
        self._byte_budget -= cost
