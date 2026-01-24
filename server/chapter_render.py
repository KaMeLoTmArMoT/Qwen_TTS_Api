# chapter_render.py
import io
import time
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Span:
    kind: str  # "narr" | "dialog" | "pause"
    text: str  # for pause can be "" (or any marker)
    pause_ms: int = 0  # used when kind == "pause"


def _silence(sr: int, ms: int) -> np.ndarray:
    n = int(sr * ms / 1000)
    return np.zeros(n, dtype=np.float32)


def _fade_in_out(w: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32)
    k = int(sr * fade_ms / 1000)
    if k <= 1 or len(w) < 2 * k:
        return w
    ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
    w[:k] *= ramp
    w[-k:] *= ramp[::-1]
    return w


def render_chapter_customvoice(
        model,
        spans: List[Span],
        language: str,
        speaker_map: Dict[str, str],
        instruct_map: Dict[str, str],
        lead_in_ms: int = 600,
        gap_ms: int = 60,
        default_pause_ms: int = 450,
        fade_ms: int = 18,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        batch_size: int = 8,
) -> tuple[bytes, int]:
    # Prepare from non-pause spans
    idxs, texts, speakers, instructs, languages = [], [], [], [], []
    for i, s in enumerate(spans):
        if s.kind == "pause":
            continue
        t = (s.text or "").strip()
        if not t:
            continue
        idxs.append(i)
        texts.append(t)
        speakers.append(speaker_map.get(s.kind, speaker_map.get("narr", "Ryan")))
        instructs.append(instruct_map.get(s.kind, ""))
        languages.append(language)

    total = len(spans)
    to_gen = len(texts)
    print(f"Total spans: {total}, to generate: {to_gen}, batch_size: {batch_size}")

    # Generate in batches, map back to original indices
    idx_to_wav: Dict[int, np.ndarray] = {}
    sr: int | None = None

    if to_gen > 0:
        n_batches = (to_gen + batch_size - 1) // batch_size
        print(f"Generate start: batches={n_batches}")

        t0 = time.perf_counter()
        for b, start in enumerate(range(0, to_gen, batch_size), start=1):
            t1 = time.perf_counter()
            end = min(start + batch_size, to_gen)
            print(f"Batch {b}/{n_batches}: items {start}..{end - 1} (count={end - start})")

            wavs_tmp, sr_tmp = model.generate_custom_voice(
                text=texts[start:end],
                language=languages[start:end],
                speaker=speakers[start:end],
                instruct=instructs[start:end],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

            if sr is None:
                sr = sr_tmp
            elif sr != sr_tmp:
                raise ValueError(f"Sample rate changed across batches: {sr} -> {sr_tmp}")

            for j, w in enumerate(wavs_tmp):
                idx_to_wav[idxs[start + j]] = w
            print(f" Batch {b} done in {time.perf_counter() - t1:.2f}s")

        print("Generate done. Total time:", time.perf_counter() - t0)

    # Concat
    if sr is None:
        raise ValueError("No audio generated (no non-pause spans)")

    out = []
    if lead_in_ms > 0:
        out.append(_silence(sr, lead_in_ms))

    for i, s in enumerate(spans):
        if s.kind == "pause":
            out.append(_silence(sr, s.pause_ms or default_pause_ms))
            continue

        if gap_ms > 0:
            out.append(_silence(sr, gap_ms))

        w = idx_to_wav.get(i)
        if w is None:
            continue
        out.append(_fade_in_out(w, sr, fade_ms))

    final = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    buf = io.BytesIO()
    sf.write(buf, final, sr, format="WAV")
    return buf.getvalue(), sr
