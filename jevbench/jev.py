from __future__ import annotations

import hashlib
import json
import math
import os
import random
import urllib.error
import urllib.request

from .questions import QUESTIONS


def _choice_confidence(probs: dict) -> float:
    """1 - normalised entropy. Only used if a backend omits the confidence field."""
    ps = [p for p in probs.values() if p > 0]
    if len(probs) < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in ps)
    return 1.0 - h / math.log(len(probs))


def _normalise(answers: dict) -> dict:
    out = {}
    for name, a in answers.items():
        a = dict(a)
        if "probabilities" in a:
            a["probabilities"] = {str(k): float(v) for k, v in a["probabilities"].items()}
        if a.get("type") in ("choice", "score") and a.get("confidence") is None:
            a["confidence"] = _choice_confidence(a["probabilities"])
        out[name] = a
    return out


class TypeSafeJev:
    name = "jev"

    def __init__(self, model: str = "jev-1.13.0", api_key: str | None = None):
        from typesafe_sdk import TypeSafeClient

        self.client = TypeSafeClient(model=model, api_key=api_key or os.environ.get("TYPESAFE_API_KEY"))
        self.model = model

    def ask(self, text: str):
        r = self.client.system_one(state=text, questions=QUESTIONS)
        answers = {k: v.model_dump(mode="json") for k, v in r.answers.items()}
        usage = {"input_tokens": r.usage.input_tokens or 0, "output_tokens": r.usage.output_tokens or 0}
        return _normalise(answers), usage, r.model

    def close(self):
        self.client.close()


class CloudflareJev:
    name = "jev"

    def __init__(self, model: str = "typesafe/jev", account_id: str | None = None, token: str | None = None):
        self.account_id = account_id or os.environ["CLOUDFLARE_ACCOUNT_ID"]
        self.model = model
        self.token = token or os.environ["CLOUDFLARE_API_TOKEN"]
        base = os.environ.get("CLOUDFLARE_API_BASE", "https://api.cloudflare.com/client/v4")
        self.url = f"{base}/accounts/{self.account_id}/ai/run"

    def ask(self, text: str):
        body = {"model": self.model, "input": {"state": text, "questions": QUESTIONS}}
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Cloudflare HTTP {e.code}: {e.read()[:300]!r}") from None
        if data.get("success") is False:
            raise RuntimeError(f"Cloudflare error: {data.get('errors')}")
        result = data.get("result", data)
        usage = result.get("usage") or {}
        usage = {"input_tokens": usage.get("input_tokens") or 0, "output_tokens": usage.get("output_tokens") or 0}
        return _normalise(result["answers"]), usage, result.get("model", self.model)

    def close(self):
        pass


class FakeJev:
    name = "jev"
    model = "FAKE-jev (dry run)"
    KEYWORDS = {
        "billing": ["charg", "refund", "pay", "invoice", "promo", "списали", "деньги", "оплат", "points", "₸"],
        "delivery": ["parcel", "courier", "deliver", "order #", "tracking", "посылк", "курьер", "заказ", "arrive"],
        "technical": ["app", "crash", "error", "load", "button", "приложени", "сайт", "sms", "ошибк"],
        "account_security": ["someone", "hack", "logged", "взлом", "не делал", "не я", "scam", "cvv", "код"],
        "general": ["?", "policy", "thank", "спасибо", "recommend", "подскажите"],
        "not_a_support_request": ["!!!", "click", "subscribe", "займы", "seo", "crypto", "test", "продам"],
    }

    def ask(self, text: str):
        rnd = random.Random(int(hashlib.md5(text.encode()).hexdigest(), 16))
        low = text.lower()
        raw = {k: 0.3 + sum(w in low for w in ws) + rnd.random() for k, ws in self.KEYWORDS.items()}
        z = sum(math.exp(2 * v) for v in raw.values())
        probs = {k: math.exp(2 * v) / z for k, v in raw.items()}
        top = max(probs, key=probs.get)
        p_sec = min(0.98, 0.05 + 0.9 * probs["account_security"] + 0.1 * rnd.random())
        urg = 2.6 if top == "account_security" else (1.8 if any(w in low for w in ["today", "tomorrow", "срочно", "завтра"]) else 0.9)
        answers = {
            "topic": {"type": "choice", "choice": top, "probabilities": probs},
            "urgency": {"type": "score", "score": urg, "probabilities": {"0": .1, "1": .4, "2": .3, "3": .2}},
            "unauthorized_access": {"type": "noul", "noul": p_sec},
            "injection_attempt": {"type": "noul", "noul": 0.9 if ("ignore" in low or "system" in low) else 0.03},
        }
        return _normalise(answers), {"input_tokens": 700 + len(text) // 3, "output_tokens": 12}, self.model

    def close(self):
        pass


def make_jev(backend: str, model: str | None = None):
    if backend == "typesafe":
        return TypeSafeJev(model=model or "jev-1.13.0")
    if backend == "cloudflare":
        return CloudflareJev(model=model or "typesafe/jev")
    if backend == "fake":
        return FakeJev()
    raise ValueError(f"unknown Jev backend {backend!r}")
