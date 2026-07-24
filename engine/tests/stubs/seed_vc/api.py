"""Fake seed_vc.api — mirrors create_v1_stream_state / the stream state."""

import numpy as np

# deliberately hot: 1.7 is past full scale, so the worker's peak guard has to
# pull the reply back under ±1.0 (a real vocoder overshoots on hot references)
HOT_LEVEL = 1.7
SAMPLE_RATE = 22050


class _StreamState:
    def __init__(self, f0_condition=False, fp16=False, realtime=False):
        self.sr = SAMPLE_RATE
        self.target_name = None
        self.f0_condition = f0_condition
        self.fp16 = fp16
        self.realtime = realtime
        self.calls = []

    def prepare_target(self, f0, target, name):
        self.target_name = name

    def process_chunk(
        self,
        source,
        length_adjust=1.0,
        diffusion_steps=25,
        inference_cfg_rate=0.7,
        f0_condition=False,
        auto_f0_adjust=True,
        semi_tone_shift=0,
        fp16_flag=False,
        end_of_stream=False,
    ):
        self.calls.append(
            {"steps": diffusion_steps, "f0": f0_condition, "eos": end_of_stream}
        )
        n = len(source.samples)
        return np.ones(n, dtype=np.float32) * HOT_LEVEL


def create_v1_stream_state(
    target=None, new_target_name=None, f0_condition=False, fp16=False, realtime=False
):
    return _StreamState(f0_condition=f0_condition, fp16=fp16, realtime=realtime)
