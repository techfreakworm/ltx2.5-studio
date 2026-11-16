# Picture-check fixtures

Real frames for `tests/test_frame_structure.py`, all in `frames.json` (text only: the repository
carries no binary files). Each entry is one decoded frame, downscaled to the check's analysis size
(long side 256 px, area-averaged), saved as JPEG quality 90 and stored base64-encoded with its
label and source. The name starts with the label: `good` (a real, usable picture), `bad_grey`
(flat grey, no picture), `bad_washed` (a picture under a grey veil).

Sources are the lead's own local real-weights runs of LTX-2.5 (distilled, Q8 GGUF, Apple MPS,
2026-11) and the operator's own failing output; every source was looked at by eye and labelled
before any statistic was measured.

| name | source | frame | what it shows |
|---|---|---|---|
| good_1 | local run G1, 512x320x25 | 12 of 25 | street at dusk with a tram |
| good_2 | local run G2, 704x480x49 | 24 of 49 | tram in a street |
| good_3 | local run G3, 704x480x121 | 60 of 121 | tram close-up |
| good_4 | local run M02 (image to video), 704x480x121 | 60 of 121 | tram in sunlight |
| good_5 | local run M03 (image and text to video), 704x480x121 | 60 of 121 | tram in sunlight |
| good_6 | local run M04 (video to video), 704x480x121 | 60 of 121 | snowy cabin under an aurora, night |
| good_7 | local run M05 (text to audio and video), 704x480x121 | 60 of 121 | ship at sea: low-texture sky and water |
| good_8 | local run M05, same clip | 0 of 121 | ship at sea, first frame |
| good_9 | local run M06 (image to audio and video), 704x480x121 | 60 of 121 | dark night scene with an aurora |
| good_10 | local run M07 (image and text to audio and video), 480x704x121 | 60 of 121 | dark portrait |
| good_11 | local run G0b, 256x256x9 | 4 of 9 | soft, painterly street at 256x256 |
| good_12 | local run X3a (text to audio and video), 49 frames | 24 of 49 | overcast beach with a dog: naturally low in contrast, lifted blacks, little colour |
| good_13 | local run X3a, same clip | 0 of 49 | overcast beach, first frame |
| good_14 | local run X2a (text to video), 49 frames | 24 of 49 | black-and-white noir street: no colour at all |
| bad_grey_1 | the operator's failing output, 512x960x121 | 0 of 121 | flat grey |
| bad_grey_2 | the operator's failing output | 60 of 121 | flat grey |
| bad_washed_1 | local run X4, the audio-to-video path before its fix, 704x480x113 | 0 of 113 | ship under a grey veil |
| bad_washed_2 | local run X4 | 56 of 113 | ship under a grey veil |
| bad_washed_3 | local run X4 | 112 of 113 | ship under a grey veil |

There is no colour-static fixture: the only real static frames are Space outputs that are not
copied into this repository. The tests synthesise blobby colour static instead and check that it
measures like those frames did.
