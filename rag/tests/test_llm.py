"""Tests for per-request LLM routing (TZ-07)."""

from __future__ import annotations

import pytest

from rag.llm import (
    CompletionOptions,
    LLMError,
    LLMRouter,
    OpenAICompatProvider,
    OllamaProvider,
    build_router_from_env,
)


def make_ollama(calls: list[tuple[str, dict]]):
    """Ollama provider backed by an in-memory transport, recording calls."""

    def transport(url: str, body: dict) -> dict:
        calls.append((url, body))
        return {'message': {'role': 'assistant', 'content': 'ok'}}

    return OllamaProvider(
        base_url='http://fake:11434',
        default_model='deepseek-r1:8b',
        transport=transport,
    )


def test_ollama_model_is_per_request():
    """The requested model lands in the request body, not a client default."""
    calls: list[tuple[str, dict]] = []
    out = make_ollama(calls).complete(
        'hi', CompletionOptions(model='qwen2.5:7b'),
    )
    assert out == 'ok'
    url, body = calls[0]
    assert url == 'http://fake:11434/api/chat'
    assert body['model'] == 'qwen2.5:7b'
    assert body['stream'] is False
    assert body['messages'][-1] == {'role': 'user', 'content': 'hi'}


def test_ollama_default_model_without_override():
    """Without an override the provider default model is used."""
    calls: list[tuple[str, dict]] = []
    make_ollama(calls).complete('hi', CompletionOptions())
    assert calls[0][1]['model'] == 'deepseek-r1:8b'


def test_ollama_system_and_temperature_in_body():
    """System message and sampling options reach the request body."""
    calls: list[tuple[str, dict]] = []
    make_ollama(calls).complete(
        'hi', CompletionOptions(system='sys', temperature=0.2),
    )
    _, body = calls[0]
    assert body['messages'][0] == {'role': 'system', 'content': 'sys'}
    assert body['options']['temperature'] == 0.2


def test_ollama_transport_error_wrapped():
    """Transport failures are wrapped into LLMError."""
    def boom(url: str, body: dict) -> dict:
        raise ConnectionError('down')

    provider = OllamaProvider(transport=boom)
    with pytest.raises(LLMError, match='request failed'):
        provider.complete('hi', CompletionOptions())


def test_ollama_malformed_response_raises():
    """A response without content raises LLMError."""
    def transport(url: str, body: dict) -> dict:
        return {'error': 'boom'}

    provider = OllamaProvider(transport=transport)
    with pytest.raises(LLMError, match='no content'):
        provider.complete('hi', CompletionOptions())


def test_openai_model_per_request():
    """OpenAI-compatible backend routes the model per request too."""
    calls: list[tuple[str, dict]] = []

    def transport(url: str, body: dict) -> dict:
        calls.append((url, body))
        return {'choices': [{'message': {'content': 'done'}}]}

    provider = OpenAICompatProvider(
        base_url='http://fake/v1', default_model='m1',
        api_key='secret', transport=transport,
    )
    out = provider.complete('hi', CompletionOptions(model='m2'))
    assert out == 'done'
    url, body = calls[0]
    assert url == 'http://fake/v1/chat/completions'
    assert body['model'] == 'm2'


def test_router_default_provider_used_without_override():
    """No explicit provider means the LLM_PROVIDER default is used."""
    calls: list[tuple[str, dict]] = []
    router = LLMRouter(default_provider='ollama')
    router.register('ollama', make_ollama(calls))
    router.complete('hi')
    assert calls[0][0].startswith('http://fake:11434')


def test_router_per_request_provider_override():
    """Same worker, two requests pinned to different providers."""
    ollama_calls: list[tuple[str, dict]] = []
    openai_calls: list[tuple[str, dict]] = []
    router = LLMRouter(default_provider='ollama')
    router.register('ollama', make_ollama(ollama_calls))

    def transport(url: str, body: dict) -> dict:
        openai_calls.append((url, body))
        return {'choices': [{'message': {'content': 'x'}}]}

    router.register('vllm', OpenAICompatProvider(
        base_url='http://vllm/v1', default_model='big',
        transport=transport,
    ))
    router.complete('a')
    router.complete('b', provider='vllm')
    assert len(ollama_calls) == 1
    assert openai_calls[0][0] == 'http://vllm/v1/chat/completions'


def test_router_per_request_model_routing_multitenant():
    """One Ollama worker serves tenants pinned to different models."""
    calls: list[tuple[str, dict]] = []
    router = LLMRouter()
    router.register('ollama', make_ollama(calls))
    router.complete('q', options=CompletionOptions(model='tenant-a-model'))
    router.complete('q', options=CompletionOptions(model='tenant-b-model'))
    models = {body['model'] for _, body in calls}
    assert models == {'tenant-a-model', 'tenant-b-model'}


def test_router_unknown_provider_error_lists_registered():
    """Unknown provider raises LLMError with the registered names."""
    router = LLMRouter()
    router.register('ollama', make_ollama([]))
    with pytest.raises(LLMError, match="Unknown LLM provider 'gpt'"):
        router.complete('hi', provider='gpt')


def test_router_async_completion():
    """Async path works through an injected async transport."""
    import asyncio

    async def run() -> str:
        async def atransport(url: str, body: dict) -> dict:
            return {'message': {'content': 'async-ok'}}

        provider = OllamaProvider(async_transport=atransport)
        return await provider.acomplete('hi', CompletionOptions())

    assert asyncio.run(run()) == 'async-ok'


def test_openai_async_completion_uses_async_transport():
    """Async OpenAI path goes through the injected async transport."""
    import asyncio

    async def run() -> str:
        calls: list[tuple[str, dict]] = []

        async def atransport(url: str, body: dict) -> dict:
            calls.append((url, body))
            return {'choices': [{'message': {'content': 'async-done'}}]}

        provider = OpenAICompatProvider(
            base_url='http://fake/v1', default_model='m1',
            api_key='secret', async_transport=atransport,
        )
        out = await provider.acomplete('hi', CompletionOptions(model='m2'))
        assert calls[0][0] == 'http://fake/v1/chat/completions'
        assert calls[0][1]['model'] == 'm2'
        return out

    assert asyncio.run(run()) == 'async-done'


def test_openai_async_transport_error_wrapped():
    """Async transport failures are wrapped into LLMError as well."""
    import asyncio

    async def boom(url: str, body: dict) -> dict:
        raise ConnectionError('down')

    async def run() -> None:
        provider = OpenAICompatProvider(
            base_url='http://fake/v1', default_model='m1',
            async_transport=boom,
        )
        await provider.acomplete('hi', CompletionOptions())

    with pytest.raises(LLMError, match='async request failed'):
        asyncio.run(run())


def test_build_router_from_env_selects_default():
    """LLM_PROVIDER sets the default provider of the router."""
    router = build_router_from_env({'LLM_PROVIDER': 'ollama'})
    assert router.default_provider == 'ollama'
    assert router.get().name == 'ollama'


def test_build_router_openai_registered_only_with_base_url():
    """The openai provider appears only when LLM_BASE_URL is set."""
    router = build_router_from_env({'LLM_BASE_URL': 'http://x/v1'})
    assert router.get('openai').name == 'openai'

    bare = build_router_from_env({})
    with pytest.raises(LLMError, match='Unknown LLM provider'):
        bare.get('openai')
