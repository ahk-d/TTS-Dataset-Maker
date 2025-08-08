import os, json, random, tempfile
import numpy as np
import soundfile as sf
import torch

# expects: from df import enhance, init_df
try:
    from df import enhance, init_df
except Exception:
    enhance = init_df = None

class TTSDenoisePipeline:
    def __init__(self, base="/content/TTS-Dataset-Maker"):
        self.base = base.rstrip("/")
        self.meta_orig = f"{self.base}/output/segments/metadata.json"
        self.meta_denoised = f"{self.base}/output/segments/denoised_metadata.json"
        self.out_dir = f"{self.base}/output/segments/audio_denoised"
        self.model = None
        self.df_state = None
        self.metadata = None

    # ---------- helpers ----------
    @staticmethod
    def _safe_load_json(p):
        if not os.path.exists(p): return None
        try:
            with open(p, "r") as f: return json.load(f)
        except Exception: return None

    @staticmethod
    def _ensure_mono_float32(x):
        if x.ndim == 1: x = x.astype(np.float32)[None, :]
        elif x.ndim == 2: x = x.mean(axis=1).astype(np.float32)[None, :]
        else: raise ValueError(f"Unsupported audio ndim: {x.ndim}")
        return x

    @staticmethod
    def _to_pcm16_wav(src_path):
        data, sr = sf.read(src_path, always_2d=False)
        if getattr(data, "dtype", None) is not None and data.dtype.kind == "f":
            data = np.nan_to_num(data)
            data = np.clip(data, -1.0, 1.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, data, int(sr), subtype="PCM_16")
        return tmp.name, sr

    # ---------- steps ----------
    def load_metadata(self):
        self.metadata = self._safe_load_json(self.meta_orig) or {"segments": []}
        return len(self.metadata.get("segments", []))

    def init_df(self):
        if init_df is None:
            raise RuntimeError("DeepFilterNet not available (missing df.init_df).")
        self.model, self.df_state, _ = init_df()
        return True

    def denoise_all(self):
        if not (self.metadata and self.model is not None and self.df_state is not None):
            raise RuntimeError("Missing metadata/model/state.")
        os.makedirs(self.out_dir, exist_ok=True)

        segs = self.metadata.get("segments", [])
        for i, seg in enumerate(segs):
            seg_id = seg.get("id", i)
            in_path = os.path.join(self.base, seg["audio_path"])
            out_path = os.path.join(self.out_dir, f"segment_{seg_id:06d}_denoised.wav")
            try:
                audio, sr = sf.read(in_path)
                audio = self._ensure_mono_float32(audio)
                with torch.no_grad():
                    enhanced = enhance(self.model, self.df_state, torch.from_numpy(audio))
                sf.write(out_path, enhanced.squeeze().cpu().numpy(), sr)
            except FileNotFoundError:
                # skip silently to keep it lean
                continue
            except Exception:
                continue
        return True

    def write_denoised_metadata(self):
        if not self.metadata: raise RuntimeError("No metadata loaded.")
        den = json.loads(json.dumps(self.metadata))  # cheap deep copy
        for i, seg in enumerate(den.get("segments", [])):
            seg_id = seg.get("id", i)
            rel = f"output/segments/audio_denoised/segment_{seg_id:06d}_denoised.wav"
            full = os.path.join(self.base, rel)
            if os.path.exists(full): seg["audio_path"] = rel
        with open(self.meta_denoised, "w") as f:
            json.dump(den, f, indent=2)
        return self.meta_denoised

    def preview_random(self):
        """
        Notebook-friendly: returns (seg_id, transcript, speaker, orig_wav, orig_sr, den_wav, den_sr).
        In notebooks, you can do:
            from IPython.display import Audio, display
            seg_id, txt, spk, o, osr, d, dsr = pipe.preview_random()
            display(Audio(filename=o))
            display(Audio(filename=d))
        """
        orig = self._safe_load_json(self.meta_orig) or {"segments": []}
        den  = self._safe_load_json(self.meta_denoised) or {"segments": []}
        om = {s.get("id", i): s for i, s in enumerate(orig.get("segments", []))}
        dm = {s.get("id", i): s for i, s in enumerate(den.get("segments", []))}
        common = list(set(om) & set(dm))
        if not common: raise RuntimeError("No common segment IDs to preview.")
        seg_id = random.choice(common)
        o, d = om[seg_id], dm[seg_id]
        o_path = os.path.join(self.base, o["audio_path"])
        d_path = os.path.join(self.base, d["audio_path"])
        o_safe, osr = self._to_pcm16_wav(o_path)
        d_safe, dsr = self._to_pcm16_wav(d_path)
        return seg_id, o.get("text",""), o.get("speaker","N/A"), o_safe, osr, d_safe, dsr

    # ---------- one-shot convenience ----------
    def run(self, preview=False):
        self.load_metadata()
        self.init_df()
        self.denoise_all()
        out_meta = self.write_denoised_metadata()
        if preview:
            return {"denoised_metadata": out_meta, "preview": self.preview_random()}
        return {"denoised_metadata": out_meta}
