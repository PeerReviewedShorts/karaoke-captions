"""Optional: an LLM proofreads the transcript for runs WITHOUT an agent (cron, CI, batch jobs). Inside Claude Code or
Codex this is not needed — the agent reads clip.review.txt and clip.disagree.json itself and files suggestions with
`karaoke-agent suggest`.

The model returns a list of suggestions (exact old words + their word indices + new text + reason), never a rewritten
transcript. Suggestions land in clip.suggestions.json as "open"; a human or an agent accepts or rejects them.

Needs `pip install anthropic` and credentials (ANTHROPIC_API_KEY or an `ant auth login` profile).
"""
import json

from karaoke import words as wd

MODEL = "claude-opus-5-5"
SYSTEM = """You proofread automatic speech recognition output that will be burned into a video as karaoke captions.

Suggest a correction only where the transcript most likely does NOT match what was said: misheard names and brands
(check the glossary), wrong numbers, wrong word forms or tenses, homophones, words split or merged wrongly. Two
recognizers disagreeing on a word is a strong hint; both agreeing does not prove the word is right.

Do not change style, wording, grammar of the speaker, filler words, or punctuation unless the meaning changes. Never
paraphrase. Each suggestion quotes the exact transcript words it replaces and their word indices [i0, i1) as numbered in
the transcript. If you are not sure, still suggest it and say why in `why`; a human decides. Return an empty list when
nothing needs fixing."""

SCHEMA = {
    "type": "object",
    "properties": {"suggestions": {"type": "array", "items": {
        "type": "object",
        "properties": {"i0": {"type": "integer"}, "i1": {"type": "integer"}, "old": {"type": "string"},
                       "new": {"type": "string"}, "why": {"type": "string"}},
        "required": ["i0", "i1", "old", "new", "why"], "additionalProperties": False}}},
    "required": ["suggestions"], "additionalProperties": False,
}


def build_prompt(words, disagree=None, glossary=None, topic=None):
    numbered = " ".join(f"[{i}]{w['w']}" for i, w in enumerate(words))
    parts = [f"Transcript (word index in brackets):\n{numbered}"]
    if topic:
        parts.append(f"Topic of the video: {topic}")
    if glossary:
        parts.append("Glossary (correct spellings):\n" + "\n".join(
            f"- {g['term']}" + (f" ({g['note']})" if g.get("note") else "") for g in glossary))
    if disagree:
        parts.append("Where a second recognizer (Whisper) heard something else:\n" + "\n".join(
            f"- [{d['i0']},{d['i1']}) parakeet: {d['parakeet']!r} / whisper: {d['whisper']!r}" for d in disagree
            if d["kind"] == "words"))
    return "\n\n".join(parts)


def run(words, disagree=None, glossary=None, topic=None, model=MODEL, effort="medium"):
    """Returns a list of {"i0", "i1", "old", "new", "why"} checked against the transcript."""
    try:
        import anthropic  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        raise RuntimeError("the LLM proofreader needs `pip install anthropic`") from e
    client = anthropic.Anthropic()
    try:
        r = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(words, disagree, glossary, topic)}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                       # on a policy decline the API re-runs on its recommended fallback
        )
    except anthropic.AuthenticationError as e:
        raise RuntimeError("no valid Anthropic credentials (set ANTHROPIC_API_KEY or run `ant auth login`)") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"rate limited by the API, retry later ({e.message})") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError("could not reach the Anthropic API") from e
    if r.stop_reason == "refusal":
        raise RuntimeError("the model declined to proofread this transcript")
    if r.stop_reason == "max_tokens":
        raise RuntimeError("the answer was cut off (max_tokens); proofread a shorter clip")
    text = next((b.text for b in r.content if b.type == "text"), "")
    out = []
    for s in json.loads(text).get("suggestions", []):
        i0, i1 = s["i0"], s["i1"]
        here = " ".join(w["w"] for w in words[i0:i1]) if 0 <= i0 <= i1 <= len(words) else ""
        if wd.norm(here) != wd.norm(s["old"]):          # indices off: look the words up instead
            i0 = wd.find(words, s["old"])
            if i0 < 0:
                continue
            i1 = i0 + len([x for x in s["old"].split() if wd.norm(x)])
        out.append({"i0": i0, "i1": i1, "old": " ".join(w["w"] for w in words[i0:i1]), "new": s["new"], "why": s["why"]})
    return out
