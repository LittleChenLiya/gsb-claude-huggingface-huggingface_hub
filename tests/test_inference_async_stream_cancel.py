# coding=utf-8
# Copyright 2025-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for cancellation propagation in streamed async inference requests.

These tests use a fake HTTP stream (httpx2.MockTransport + custom AsyncByteStream) to verify
that:
- cancelling an async inference request (before or after the first token) immediately closes
  the underlying stream and releases the connection,
- the close procedure runs exactly once,
- the retry logic never re-sends a request whose stream has started,
- normal completion and mid-stream errors also close the stream.
"""

import asyncio
import json

import httpx2
import pytest

from huggingface_hub import AsyncInferenceClient
from huggingface_hub.inference._common import GenerationError, RequestParameters


pytestmark = pytest.mark.inference

FAKE_BASE_URL = "http://fake-inference.local"
TGI_TOKEN_PAYLOAD = {
    "index": 0,
    "token": {"id": 0, "text": "hello", "logprob": 0.0, "special": False},
}


def _sse_line(payload: dict) -> bytes:
    return ("data: " + json.dumps(payload) + "\n\n").encode()


def _request_parameters(url: str) -> RequestParameters:
    return RequestParameters(url=url, task="text-generation", model=None, json={}, data=None, headers={})


class FakeSSEByteStream(httpx2.AsyncByteStream):
    """A controllable SSE response body.

    Tracks how many times it is closed (`close_count`) which asserts the underlying HTTP
    connection is released exactly once. `entered` is set as soon as iteration starts and
    `release` gates the delivery of the first chunk, allowing cancellation *before the first
    token*.
    """

    def __init__(
        self,
        chunks,
        entered: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
        block_before_first: bool = False,
        block_after_first: bool = False,
    ):
        self._chunks = list(chunks)
        self._entered = entered
        self._release = release
        self._block_before_first = block_before_first
        self._block_after_first = block_after_first
        self.close_count = 0

    async def __aiter__(self):
        if self._entered is not None:
            self._entered.set()
        if self._block_before_first and self._release is not None:
            await self._release.wait()
        for i, chunk in enumerate(self._chunks):
            yield chunk
            if i == 0 and self._block_after_first and self._release is not None:
                # Block right after the first chunk was delivered, so cancellation can happen
                # while the stream is open and after the first token was streamed.
                await self._release.wait()

    async def aclose(self) -> None:
        self.close_count += 1


def _make_client(handler) -> AsyncInferenceClient:
    """Create an AsyncInferenceClient wired to a fake HTTP transport."""
    client = AsyncInferenceClient(model=FAKE_BASE_URL + "/models/gpt2")
    # Same response event hook as the production client: error bodies are eagerly read.
    from huggingface_hub.utils._http import async_hf_response_event_hook

    client._async_client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        base_url=FAKE_BASE_URL,
        event_hooks={"response": [async_hf_response_event_hook]},
    )
    return client


@pytest.mark.asyncio
async def test_stream_cancelled_before_first_token_closes_connection():
    """Cancellation while waiting for the first token closes the stream at once."""
    request_count = 0
    streams: list[FakeSSEByteStream] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        stream = FakeSSEByteStream(
            [_sse_line(TGI_TOKEN_PAYLOAD), b"data: [DONE]\n\n"],
            entered=entered,
            release=release,
            block_before_first=True,
        )
        streams.append(stream)
        return httpx2.Response(200, stream=stream, request=request)

    client = _make_client(handler)
    stream = await client.text_generation("test", stream=True, max_new_tokens=1)
    tokens = []

    async def consume():
        async for token in stream:
            tokens.append(token)

    task = asyncio.create_task(consume())
    await entered.wait()  # the response arrived but no token has been yielded yet

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tokens == []  # cancelled before the first token
    assert request_count == 1  # no retry
    assert streams[0].close_count == 1  # closed exactly once

    await client._async_client.aclose()


@pytest.mark.asyncio
async def test_stream_cancelled_while_request_in_flight():
    """Cancellation while the POST request is still in flight is propagated to the transport."""
    request_count = 0
    streams: list[FakeSSEByteStream] = []
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        handler_started.set()
        await release_handler.wait()  # simulate a slow server not sending headers yet
        stream = FakeSSEByteStream([_sse_line(TGI_TOKEN_PAYLOAD), b"data: [DONE]\n\n"])
        streams.append(stream)
        return httpx2.Response(200, stream=stream, request=request)

    client = _make_client(handler)

    async def consume():
        stream = await client.text_generation("test", stream=True, max_new_tokens=1)
        async for _ in stream:
            pass

    task = asyncio.create_task(consume())
    await handler_started.wait()  # the request reached the (fake) server

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release_handler.set()  # let the handler finish; no one should consume the response

    assert request_count == 1  # a cancelled in-flight request is never re-sent
    assert streams == []  # the response was never built by the handler

    await client._async_client.aclose()


@pytest.mark.asyncio
async def test_stream_cancelled_after_first_token_closes_connection():
    """Cancellation mid-stream (after some tokens) closes the stream at once."""
    request_count = 0
    streams: list[FakeSSEByteStream] = []
    stream_entered = asyncio.Event()
    first_token = asyncio.Event()
    release_remainder = asyncio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        chunks = [
            _sse_line({"index": i, "token": {"id": i, "text": str(i), "logprob": 0.0, "special": False}})
            for i in range(10)
        ]
        # First chunk is delivered immediately; the rest are gated until the test releases them.
        stream = FakeSSEByteStream(chunks, entered=stream_entered, release=release_remainder, block_after_first=True)
        streams.append(stream)
        return httpx2.Response(200, stream=stream, request=request)

    client = _make_client(handler)
    stream = await client.text_generation("test", stream=True, max_new_tokens=10)

    received = []

    async def consume():
        async for token in stream:
            received.append(token)
            first_token.set()

    task = asyncio.create_task(consume())
    await first_token.wait()  # at least one token was streamed and yielded

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert request_count == 1  # the started stream is never re-sent
    assert len(received) >= 1
    assert streams[0].close_count == 1  # closed exactly once

    await client._async_client.aclose()


@pytest.mark.asyncio
async def test_stream_fully_consumed_closes_connection():
    """Fully consuming a stream closes it when iteration ends."""
    request_count = 0
    streams: list[FakeSSEByteStream] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        stream = FakeSSEByteStream([_sse_line(TGI_TOKEN_PAYLOAD), b"data: [DONE]\n\n"])
        streams.append(stream)
        return httpx2.Response(200, stream=stream, request=request)

    client = _make_client(handler)
    stream = await client.text_generation("test", stream=True, max_new_tokens=1)
    tokens = [token async for token in stream]

    assert tokens == ["hello"]
    assert request_count == 1
    assert streams[0].close_count == 1

    await client._async_client.aclose()


@pytest.mark.asyncio
async def test_chat_completion_stream_cancelled_after_first_chunk():
    """Cancelling a streamed chat completion releases the connection (covers the chat parser)."""
    request_count = 0
    streams: list[FakeSSEByteStream] = []
    first_chunk = asyncio.Event()
    release_remainder = asyncio.Event()

    chat_chunk = {
        "id": "id",
        "model": "gpt2",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}],
    }

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        stream = FakeSSEByteStream(
            [_sse_line(chat_chunk), b'data: {"id": "id", "model": "gpt2", "choices": []}\n\n', b"data: [DONE]\n\n"],
            release=release_remainder,
            block_after_first=True,
        )
        streams.append(stream)
        return httpx2.Response(200, stream=stream, request=request)

    client = _make_client(handler)
    messages = [{"role": "user", "content": "Hello"}]
    stream = await client.chat_completion(messages, model="gpt2", stream=True, max_tokens=10)

    async def consume():
        async for _ in stream:
            first_chunk.set()

    task = asyncio.create_task(consume())
    await first_chunk.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert request_count == 1
    assert streams[0].close_count == 1

    await client._async_client.aclose()


@pytest.mark.asyncio
async def test_stream_closed_once_on_manual_aclose():
    """Explicitly closing the generator twice does not close the stream twice."""
    streams: list[FakeSSEByteStream] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        stream = FakeSSEByteStream([_sse_line(TGI_TOKEN_PAYLOAD), b"data: [DONE]\n\n"])
        streams.append(stream)
        return httpx2.Response(200, stream=stream, request=request)

    client = _make_client(handler)
    line_stream = await client._inner_post(_request_parameters(FAKE_BASE_URL + "/stream"), stream=True)
    first = await line_stream.__anext__()
    assert first == "data: " + json.dumps(TGI_TOKEN_PAYLOAD)

    await line_stream.aclose()
    await line_stream.aclose()  # idempotent

    assert streams[0].close_count == 1

    await client._async_client.aclose()


@pytest.mark.asyncio
async def test_error_after_started_stream_is_not_retried():
    """An SSE error arriving after the stream started must not trigger any retry."""
    request_count = 0
    streams: list[FakeSSEByteStream] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        stream = FakeSSEByteStream(
            [
                _sse_line(TGI_TOKEN_PAYLOAD),
                _sse_line({"error": "boom", "error_type": "generation"}),
            ]
        )
        streams.append(stream)
        return httpx2.Response(200, stream=stream, request=request)

    client = _make_client(handler)
    stream = await client.text_generation("test", stream=True, max_new_tokens=10)

    with pytest.raises(GenerationError, match="boom"):
        async for _ in stream:
            pass

    assert request_count == 1  # no resend of an already started stream
    assert streams[0].close_count == 1  # connection released despite the error

    await client._async_client.aclose()


@pytest.mark.asyncio
async def test_400_before_stream_starts_triggers_single_retry():
    """A 400 received before the stream starts follows the regular retry path (once)."""
    request_count = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        body = json.dumps({"error": "The following `model_kwargs` are not used by the model: ['watermark']"}).encode()
        return httpx2.Response(
            400,
            content=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            request=request,
        )

    client = _make_client(handler)
    # Retry removes `watermark` from the payload but the server still rejects the request.
    with pytest.raises(Exception):
        await client.text_generation("test", stream=True, max_new_tokens=1, watermark=True)

    assert request_count == 2  # one initial request + exactly one retry

    await client._async_client.aclose()


@pytest.mark.asyncio
async def test_http_error_before_stream_starts_closes_response():
    """A streamed response with an error status is closed before it reaches the caller."""
    request_count = 0
    streams: list[FakeSSEByteStream] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal request_count
        request_count += 1
        body = json.dumps({"error": "server failure"}).encode()
        stream = FakeSSEByteStream([body])
        streams.append(stream)
        return httpx2.Response(
            500,
            stream=stream,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            request=request,
        )

    client = _make_client(handler)
    with pytest.raises(Exception):
        await client._inner_post(_request_parameters(FAKE_BASE_URL + "/stream"), stream=True)

    assert request_count == 1
    assert streams[0].close_count == 1  # error response is closed, connection released

    await client._async_client.aclose()
