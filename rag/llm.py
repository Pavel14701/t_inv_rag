"""LLM provider layer with per-request routing (TZ-07).

Global ``LLM_PROVIDER`` environment variable only selects the *default*
provider for a worker; every single request may override both the provider
and the model. This makes multitenant scenarios possible: one Ollama worker
can serve several tenants pinned to different models at the same time.
"""

from __future__ import annotations

import os

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import niquests


class LLMError(Exception):
    """Raised on provider/routing/transport failures."""


@dataclass(frozen=True, slots=True)
class CompletionOptions:
    """Per-request generation options (optional, provider defaults)."""

    model: str | None = None
    system: str | None = None
    temperature: float | None = None
    num_predict: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def payload_for(self, prompt: str, default_model: str) -> dict[str, Any]:
        """Build a provider-agnostic request body."""
        body: dict[str, Any] = {
            'prompt': prompt,
            'model': self.model or default_model,
        }
        if self.temperature is not None:
            body['temperature'] = self.temperature
        if self.num_predict is not None:
            body['num_predict'] = self.num_predict
        body.update(self.extra)
        return body


class LLMProvider(Protocol):
    """Protocol every backend must satisfy (sync + async)."""

    name: str

    def complete(self, prompt: str, options: CompletionOptions) -> str:
        """Complete a prompt synchronously."""
        ...

    async def acomplete(self, prompt: str, options: CompletionOptions) -> str:
        """Complete a prompt asynchronously."""
        ...


Transport = Callable[[str, dict[str, Any]], dict[str, Any]]
AsyncTransport = Callable[[str, dict[str, Any]], Any]  # awaitable[dict]


class OllamaProvider:
    """Ollama ``/api/chat`` backend.

    The model is a per-request field of the Ollama API, so a single running
    worker serves any number of models — this is what enables per-request
    model routing for multitenant callers.
    """

    def __init__(
        self,
        base_url: str = 'http://localhost:11434',
        default_model: str = 'deepseek-r1:8b',
        timeout: float = 120.0,
        transport: Transport | None = None,
        async_transport: AsyncTransport | None = None,
    ) -> None:
        self.name = 'ollama'
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.timeout = timeout
        self._transport = transport
        self._async_transport = async_transport

    def _url(self) -> str:
        return f'{self.base_url}/api/chat'

    def _body(self, prompt: str, options: CompletionOptions) -> dict[str, Any]:
        body = options.payload_for(prompt, self.default_model)
        body.pop('prompt', None)
        messages: list[dict[str, str]] = []
        if options.system:
            messages.append({'role': 'system', 'content': options.system})
        messages.append({'role': 'user', 'content': prompt})
        opts = {
            k: v for k, v in body.items()
            if k in ('temperature', 'num_predict')
        }
        return {
            'model': body['model'],
            'messages': messages,
            'stream': False,
            'options': opts,
        }

    @staticmethod
    def _extract(response: dict[str, Any]) -> str:
        content = (response.get('message') or {}).get('content')
        if content is None:
            content = response.get('response')
        if not isinstance(content, str):
            raise LLMError(f'Ollama returned no content: {response!r:.200}')
        return content

    def complete(self, prompt: str, options: CompletionOptions) -> str:
        """Complete a prompt via the Ollama chat API."""
        body = self._body(prompt, options)
        try:
            if self._transport is not None:
                response = self._transport(self._url(), body)
            else:
                response = niquests.post(
                    self._url(), json=body, timeout=self.timeout,
                ).json()
        except LLMError:
            raise
        except Exception as exc:
            msg = f'Ollama request failed: {exc}'
            raise LLMError(msg) from exc
        return self._extract(response)

    async def acomplete(self, prompt: str, options: CompletionOptions) -> str:
        """Async counterpart of :meth:`complete`."""
        body = self._body(prompt, options)
        try:
            if self._async_transport is not None:
                response = await self._async_transport(self._url(), body)
            else:
                async with niquests.AsyncSession() as session:
                    resp = await session.post(
                        self._url(), json=body, timeout=self.timeout,
                    )
                    response = resp.json()
        except LLMError:
            raise
        except Exception as exc:
            msg = f'Ollama async request failed: {exc}'
            raise LLMError(msg) from exc
        return self._extract(response)


class OpenAICompatProvider:
    """OpenAI-compatible ``/chat/completions`` backend (vLLM, LM Studio, ...).

    Also takes the model per request, same routing semantics as Ollama.
    """

    def __init__(
        self,
        base_url: str,
        default_model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: Transport | None = None,
        async_transport: AsyncTransport | None = None,
    ) -> None:
        self.name = 'openai'
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.api_key = api_key
        self.timeout = timeout
        self._transport = transport
        self._async_transport = async_transport

    def _url(self) -> str:
        return f'{self.base_url}/chat/completions'

    def _body(self, prompt: str, options: CompletionOptions) -> dict[str, Any]:
        body = options.payload_for(prompt, self.default_model)
        body.pop('prompt', None)
        body.pop('num_predict', None)
        messages: list[dict[str, str]] = []
        if options.system:
            messages.append({'role': 'system', 'content': options.system})
        messages.append({'role': 'user', 'content': prompt})
        body['messages'] = messages
        return body

    @staticmethod
    def _extract(response: dict[str, Any]) -> str:
        try:
            return response['choices'][0]['message']['content'] or ''
        except (KeyError, IndexError, TypeError) as exc:
            msg = f'OpenAI-compatible response malformed: {response!r:.200}'
            raise LLMError(msg) from exc

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {'Authorization': f'Bearer {self.api_key}'}
        return {}

    def complete(self, prompt: str, options: CompletionOptions) -> str:
        """Complete a prompt via an OpenAI-compatible API."""
        body = self._body(prompt, options)
        try:
            if self._transport is not None:
                response = self._transport(self._url(), body)
            else:
                response = niquests.post(
                    self._url(), json=body, headers=self._headers(),
                    timeout=self.timeout,
                ).json()
        except LLMError:
            raise
        except Exception as exc:
            msg = f'OpenAI-compatible request failed: {exc}'
            raise LLMError(msg) from exc
        return self._extract(response)

    async def acomplete(self, prompt: str, options: CompletionOptions) -> str:
        """Async counterpart of :meth:`complete`."""
        body = self._body(prompt, options)
        try:
            if self._async_transport is not None:
                response = await self._async_transport(self._url(), body)
            else:
                async with niquests.AsyncSession() as session:
                    resp = await session.post(
                        self._url(), json=body, headers=self._headers(),
                        timeout=self.timeout,
                    )
                    response = resp.json()
        except LLMError:
            raise
        except Exception as exc:
            msg = f'OpenAI-compatible async request failed: {exc}'
            raise LLMError(msg) from exc
        return self._extract(response)


def build_router_from_env(
    env: dict[str, str] | None = None,
) -> LLMRouter:
    """Build a router configured from environment variables.

    Recognised variables:
        LLM_PROVIDER    - default provider name (default: 'ollama');
        LLM_MODEL       - default model (default: 'deepseek-r1:8b');
        OLLAMA_BASE_URL - default: http://localhost:11434;
        LLM_BASE_URL    - for the 'openai' provider;
        LLM_API_KEY     - bearer token for the 'openai' provider.
    """
    env = dict(os.environ if env is None else env)
    default_provider = env.get('LLM_PROVIDER', 'ollama')
    default_model = env.get('LLM_MODEL', 'deepseek-r1:8b')
    router = LLMRouter(default_provider=default_provider)
    router.register('ollama', OllamaProvider(
        base_url=env.get('OLLAMA_BASE_URL', 'http://localhost:11434'),
        default_model=default_model,
    ))
    if env.get('LLM_BASE_URL'):
        router.register('openai', OpenAICompatProvider(
            base_url=env['LLM_BASE_URL'],
            default_model=default_model,
            api_key=env.get('LLM_API_KEY'),
        ))
    return router


class LLMRouter:
    """Registry of named providers with per-request routing.

    The default provider (from ``LLM_PROVIDER``) is only a *fallback*:
    every call may point at any registered provider and any model.
    """

    def __init__(self, default_provider: str = 'ollama') -> None:
        self.default_provider = default_provider
        self._providers: dict[str, LLMProvider] = {}

    def register(self, name: str, provider: LLMProvider) -> None:
        """Register a backend under a routing name."""
        self._providers[name] = provider

    def get(self, name: str | None = None) -> LLMProvider:
        """Resolve a provider by name; None means the default provider."""
        key = name or self.default_provider
        provider = self._providers.get(key)
        if provider is None:
            registered = ', '.join(sorted(self._providers)) or '<none>'
            msg = f'Unknown LLM provider {key!r}. Registered: {registered}'
            raise LLMError(msg)
        return provider

    def complete(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        options: CompletionOptions | None = None,
    ) -> str:
        """Complete a prompt, routing to the requested provider."""
        return self.get(provider).complete(
            prompt, options or CompletionOptions(),
        )

    async def acomplete(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        options: CompletionOptions | None = None,
    ) -> str:
        """Async counterpart of :meth:`complete`."""
        return await self.get(provider).acomplete(
            prompt, options or CompletionOptions(),
        )
