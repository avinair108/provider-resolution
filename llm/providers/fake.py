"""A deterministic stand-in for tests and dry runs: no network, no key."""

from collections.abc import Callable

from llm.types import Request, Response


class FakeProvider:
    name = "fake"

    def __init__(self, reply: Callable[[Request], str] = lambda r: '{"ok": true}'):
        self.reply = reply
        self.calls: list[Request] = []

    def send(self, request: Request) -> Response:
        self.calls.append(request)
        text = self.reply(request)
        return Response(text=text, provider=self.name, model=request.model, input_tokens=len(request.user) // 4,
                        output_tokens=len(text) // 4, latency_s=0.0, finish_reason="stop")
