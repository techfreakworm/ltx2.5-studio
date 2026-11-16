# Where every file here comes from

Inputs (`media/`): the lead's test assets — pictures generated with the operator's codex login, speech and
sound generated with the operator's ElevenLabs key, the cabin clip a pan made from the cabin picture with
ffmpeg, the silent captain clip this app's own output for “The captain speaks” with its sound removed.

Samples (`samples/`): re-encodes of this app's own outputs from the final real-weights pass on the
operator's Mac (LTX-2.5 distilled, Q8 GGUF transformer, single pass; 2026-11-16, the code at commit
e59944e; mode 08 on ba2d582, the commit that fixed the local memory check). Each sample is what its example's
prompt, inputs, size and seed produced there:

- `samples/text_to_video.mp4` ← `evidence/local_runs/F01_t2v/` (output.mp4)
- `samples/image_to_video.mp4` ← `evidence/local_runs/F02_i2v/` (output.mp4)
- `samples/image_text_to_video.mp4` ← `evidence/local_runs/F03_it2v/` (output.mp4)
- `samples/video_to_video.mp4` ← `evidence/local_runs/F04_v2v/` (output.mp4)
- `samples/text_to_audio_video.mp4` ← `evidence/local_runs/F05_t2av/` (output.mp4)
- `samples/image_to_audio_video.mp4` ← `evidence/local_runs/F06_i2av/` (output.mp4)
- `samples/image_text_to_audio_video.mp4` ← `evidence/local_runs/F07_it2av/` (output.mp4)
- `samples/text_to_audio.m4a` ← `evidence/local_runs/F08_t2a/` (output.wav)
- `samples/video_to_audio.m4a` ← `evidence/local_runs/F09_v2a/` (output.wav)
- `samples/audio_to_audio.m4a` ← `evidence/local_runs/F10_a2a/` (output.wav)
- `samples/audio_to_video.mp4` ← `evidence/local_runs/F11b_ocean/` (output.mp4)
