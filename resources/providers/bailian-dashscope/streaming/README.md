# providers.bailian-dashscope.streaming

Cloud ASR provider script for Alibaba Bailian DashScope realtime speech recognition.

## Entry

- `entry.py`

## Runtime

- command: `python3`
- input: JSONL via stdin
- output: JSONL via stdout
- diagnostics: stderr only
- dependencies: Python standard library only

## Protocol Family

This provider uses the DashScope Recognition WebSocket API at
`/api-ws/v1/inference`. It is separate from the Bailian Qwen Omni Realtime
provider.

Qwen Omni Realtime models such as `qwen3-asr-flash-realtime` are not supported
by this provider. Use the existing Bailian Qwen streaming provider for those
models.

## Audio Format

Input audio must be mono `S16_LE` PCM at `16000 Hz`. The script sends raw audio
bytes as binary WebSocket frames and does not resample audio.

Models that require 8 kHz input need matching upstream capture audio. This first
version keeps the registry streaming contract fixed at 16 kHz PCM.

## Input Protocol

- `{"type":"audio","audio_base64":"...","commit":false}`
- `{"type":"audio","audio_base64":"...","commit":true}`
- `{"type":"finish"}`
- `{"type":"cancel"}`

`audio_base64` should contain mono `S16_LE` PCM at `16000 Hz`. The `commit`
value is accepted for compatibility with the registry streaming protocol but is
ignored because DashScope Recognition uses a continuous audio stream.

On `finish`, the script sends a DashScope `finish-task` frame and waits briefly
for final upstream results.

## Output Protocol

- `{"type":"session_started","session_id":"..."}`
- `{"type":"partial","text":"..."}`
- `{"type":"final","text":"...","segment_final":true}`
- `{"type":"error","message":"..."}`
- `{"type":"closed"}`

The emitted `session_id` is the generated DashScope `task_id`. Partial output is
the accumulated confirmed text plus the current non-final sentence. Final output
is accumulated recognized text. If the final upstream event is empty or finish
times out, the last non-empty partial is emitted as a final fallback.

## Environment Variables

- `VINPUT_ASR_API_KEY` required
  DashScope API key sent as `Authorization: bearer <api_key>`.
- `VINPUT_ASR_URL` optional
  Full DashScope WebSocket URL. Defaults to
  `wss://dashscope.aliyuncs.com/api-ws/v1/inference` and can be used to select
  another region endpoint.
- `VINPUT_ASR_MODEL` optional
  DashScope Recognition model id. Defaults to `fun-asr-realtime`. The selected
  model must support the DashScope Recognition protocol and 16 kHz PCM input.
- `VINPUT_ASR_LANGUAGES` optional
  Comma-separated language hints such as `zh` or `zh,en`. Whitespace is trimmed
  and empty items are ignored. When unset or empty, `language_hints` is omitted
  so DashScope can auto-detect when supported.
- `VINPUT_ASR_TIMEOUT` optional
  Network timeout in seconds. Defaults to `30`.
- `VINPUT_ASR_FINISH_GRACE_SECS` optional
  Extra wait time after local `finish` before the script closes the socket.
  Defaults to `1`.

## DashScope Task Envelope

The script sends a `run-task` text frame with:

- `header.action`: `run-task`
- `header.streaming`: `duplex`
- `payload.task_group`: `audio`
- `payload.task`: `asr`
- `payload.function`: `recognition`
- `payload.parameters.format`: `pcm`
- `payload.parameters.sample_rate`: `16000`
- `payload.parameters.semantic_punctuation_enabled`: `false`

Audio chunks are sent as binary WebSocket frames. On finish, the script sends a
`finish-task` text frame so DashScope can flush final recognition results.
