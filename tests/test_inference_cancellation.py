# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
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
"""Tests for inference request cancellation propagation.

These tests verify that:
1. Cancelling an async inference stream immediately closes the underlying stream
   and releases the connection (before and after the first token).
2. The close process is idempotent (runs exactly once).
3. Retry logic does not resend a stream that has already started.

A fake HTTP stream (mock) is used to count requests and track connection release,
so the tests are fully reproducible without network access.
"""

import asyncio
from unittest.mock import Mock

import pytest

from huggingface_hub import AsyncInferenceClient
from huggingface_hub.inference._common import RequestParameters


pytestmark = pytest.mark.inference


class FakeStreamResponse:
    """A fake httpx.Response that yields SSE lines and tracks its own close.

    Each call to ``aclose()`` increments ``close_count`` so tests can assert the
    stream was closed exactly once.
    """

    def __init__(self, lines):
        self._lines = list(lines)
        self.status_code = 200
        self.is_closed = False
        self.close_count = 0

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self):
        self.close_count += 1
        self.is_closed = True

    def raise_for_status(self):
        """No-op for a 200 OK response (mirrors httpx.Response.raise_for_status)."""
        pass


class FakeStreamCM:
    """A fake ``client.stream(...)`` async context manager.

    Increments ``counter["requests"]`` every time a stream is entered (i.e. every
    time an HTTP request is issued) and closes the response on exit.
    """

    def __init__(self, response, counter):
        self._response = response
        self._counter = counter
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self._counter["requests"] += 1
        self.entered = True
        return self._response

    async def __aexit__(self, *args):
        self.exited = True
        await self._response.aclose()


def make_client(lines, counter):
    """Create an AsyncInferenceClient with a fake streaming HTTP client."""
    client = AsyncInferenceClient(model="test-model")
    response = FakeStreamResponse(lines)
    mock_client = Mock()
    mock_client.stream = Mock(return_value=FakeStreamCM(response, counter))
    client._async_client = mock_client
    return client, response


def make_request_parameters():
    return RequestParameters(
        url="http://test",
        task="text-generation",
        model="test",
        json=None,
        data=None,
        headers={},
    )


@pytest.mark.asyncio
async def test_cancel_before_first_token_closes_stream():
    """Cancelling before the first token must close the stream and release the connection."""
    counter = {"requests": 0}
    lines = ["data: token1", "data: token2", "data: [DONE]"]
    client, response = make_client(lines, counter)

    gen = await client._inner_post(make_request_parameters(), stream=True)
    # Cancel before consuming any token
    await gen.aclose()

    assert counter["requests"] == 1, "Expected exactly one HTTP request"
    assert response.is_closed, "Stream should be closed after cancellation"
    assert response.close_count == 1, "Stream should be closed exactly once"


@pytest.mark.asyncio
async def test_cancel_after_first_token_closes_stream():
    """Cancelling after the first token must close the stream without resending."""
    counter = {"requests": 0}
    lines = ["data: token1", "data: token2", "data: [DONE]"]
    client, response = make_client(lines, counter)

    gen = await client._inner_post(make_request_parameters(), stream=True)
    # Consume the first token
    first = await gen.__anext__()
    assert first == "data: token1"
    # Cancel after the first token
    await gen.aclose()

    assert counter["requests"] == 1, "Retry must not resend a started stream"
    assert response.is_closed, "Stream should be closed after cancellation"
    assert response.close_count == 1, "Stream should be closed exactly once"


@pytest.mark.asyncio
async def test_close_is_idempotent():
    """Closing the stream multiple times must not error and must close exactly once."""
    counter = {"requests": 0}
    lines = ["data: token1"]
    client, response = make_client(lines, counter)

    gen = await client._inner_post(make_request_parameters(), stream=True)
    await gen.aclose()
    await gen.aclose()  # close a second time
    await gen.aclose()  # close a third time

    assert response.close_count == 1, "Stream should be closed exactly once"
    assert counter["requests"] == 1


@pytest.mark.asyncio
async def test_stream_started_flag():
    """The stream handle must track whether the first token has been yielded."""
    counter = {"requests": 0}
    lines = ["data: token1", "data: token2"]
    client, response = make_client(lines, counter)

    gen = await client._inner_post(make_request_parameters(), stream=True)
    assert gen._stream_handle.started is False, "Stream should not be started before first token"
    await gen.__anext__()
    assert gen._stream_handle.started is True, "Stream should be started after first token"
    await gen.aclose()
    assert response.is_closed


@pytest.mark.asyncio
async def test_full_consumption_closes_stream():
    """Fully consuming the stream must close it and release the connection."""
    counter = {"requests": 0}
    lines = ["data: token1", "data: token2", "data: [DONE]"]
    client, response = make_client(lines, counter)

    gen = await client._inner_post(make_request_parameters(), stream=True)
    tokens = [line async for line in gen]

    assert tokens == ["data: token1", "data: token2", "data: [DONE]"]
    assert counter["requests"] == 1
    assert response.is_closed, "Stream should be closed after full consumption"
    assert response.close_count == 1


@pytest.mark.asyncio
async def test_cancel_via_task_cancellation():
    """Cancelling the task running the stream must close the stream immediately."""
    counter = {"requests": 0}
    client = AsyncInferenceClient(model="test-model")
    response = FakeStreamResponse(["data: token1", "data: token2", "data: [DONE]"])

    async def slow_aiter_lines():
        for line in ["data: token1", "data: token2", "data: [DONE]"]:
            await asyncio.sleep(0.1)  # slow enough to be cancelled mid-stream
            yield line

    response.aiter_lines = slow_aiter_lines
    mock_client = Mock()
    mock_client.stream = Mock(return_value=FakeStreamCM(response, counter))
    client._async_client = mock_client

    gen = await client._inner_post(make_request_parameters(), stream=True)

    async def consume():
        async for _ in gen:
            pass

    task = asyncio.create_task(consume())
    # Let the task start consuming
    await asyncio.sleep(0.05)
    # Cancel the task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert counter["requests"] == 1, "Retry must not resend a cancelled stream"
    assert response.is_closed, "Stream should be closed after task cancellation"
    assert response.close_count == 1, "Stream should be closed exactly once"


@pytest.mark.asyncio
async def test_retry_guard_does_not_resend_started_stream():
    """Retry logic must not resend a stream that has already yielded a token.

    This test verifies the retry guard boundary: after a stream has started (first
    token yielded), the retry path must not issue a new HTTP request.
    """
    counter = {"requests": 0}
    lines = ["data: token1", "data: token2"]
    client, response = make_client(lines, counter)

    gen = await client._inner_post(make_request_parameters(), stream=True)
    # Consume the first token -> stream is started
    await gen.__anext__()
    assert gen._stream_handle.started is True

    # The retry guard checks `_stream_handle.started`; since the stream started,
    # no new request should be issued if a retry were attempted now.
    started = gen._stream_handle.started
    await gen.aclose()

    assert started is True
    assert counter["requests"] == 1, "No retry request must be issued after stream start"
    assert response.is_closed
    assert response.close_count == 1


@pytest.mark.asyncio
async def test_multiple_streams_share_client_without_leaking():
    """Creating and closing multiple streams on the same client must not leak connections."""
    counter = {"requests": 0}
    lines = ["data: token1"]
    client = AsyncInferenceClient(model="test-model")
    responses = []

    def make_stream_cm(*args, **kwargs):
        response = FakeStreamResponse(lines)
        responses.append(response)
        return FakeStreamCM(response, counter)

    mock_client = Mock()
    mock_client.stream = Mock(side_effect=make_stream_cm)
    client._async_client = mock_client

    for _ in range(3):
        gen = await client._inner_post(make_request_parameters(), stream=True)
        await gen.__anext__()
        await gen.aclose()

    assert counter["requests"] == 3, "Each stream issues exactly one request"
    for response in responses:
        assert response.close_count == 1, "Each stream is closed exactly once"
