import sys
from kapitsa_sdk import Engine

with Engine(sys.argv[1], context=4096, slots=1) as engine:
    for prompt in ("Say hello briefly.", "Explain Attention in one sentence."):
        result = engine.generate(
            [{"role": "user", "content": prompt}], max_tokens=16,
            temperature=0, template_kwargs={"enable_thinking": False},
            on_delta=lambda delta: print(delta, flush=True),
        )
        print(result, flush=True)
    with engine.request([{"role": "user", "content": "Count from one to ten."}],
                        max_tokens=32, template_kwargs={"enable_thinking": False}) as request:
        result = request.generate(on_delta=lambda delta: False)
        assert result["finish_reason"] == "cancelled", result
        print("Cancellation passed", flush=True)
