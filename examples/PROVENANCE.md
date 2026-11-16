# Where every file here comes from

Inputs (`media/`): the lead's test assets — pictures generated with the operator's codex login, speech and
sound generated with the operator's ElevenLabs key, the cabin clip a pan made from the cabin picture with
ffmpeg, the silent captain clip this app's own output for “The captain speaks” with its sound removed.

Samples (`samples/`): re-encodes of this app's own outputs from real-weights runs on the operator's Mac
(LTX-2.5 distilled, Q8 GGUF transformer, single pass):

- `samples/text_to_video.mp4` ← `evidence/local_runs/S01_tram_silent/` (output.mp4)
- `samples/image_to_video.mp4` ← `evidence/local_runs/M02_i2v/` (output.mp4)
- `samples/image_text_to_video.mp4` ← `evidence/local_runs/S03_tram_silent/` (output.mp4)
- `samples/video_to_video.mp4` ← `evidence/local_runs/S04_summer_l091/` (output.mp4)
- `samples/text_to_audio_video.mp4` ← `evidence/local_runs/M05_t2av/` (output.mp4)
- `samples/image_to_audio_video.mp4` ← `evidence/local_runs/M06_i2av/` (output.mp4)
- `samples/image_text_to_audio_video.mp4` ← `evidence/local_runs/M07_it2av/` (output.mp4)
- `samples/text_to_audio.m4a` ← `evidence/local_runs/M08_t2a/` (output.wav)
- `samples/video_to_audio.m4a` ← `evidence/local_runs/S09_captain_voice/` (output.wav)
- `samples/audio_to_audio.m4a` ← `evidence/local_runs/M10f_ocean_l073/` (output.wav)
- `samples/audio_to_video.mp4` ← `evidence/local_runs/M11h_a2v_firstframe/` (output.mp4)
