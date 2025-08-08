import json
import os
import numpy as np
import soundfile as sf
import pandas as pd

class DataProcessor:
    def __init__(self, json_path="output/tts_dataset.json", audio_path="output/audio.wav"):
        self.json_path = json_path
        self.audio_path = audio_path
        self.df = None
        self.full_audio = None
        self.sr = None
        self.duration_total = None
        
        self._validate_files()
        self._load_data()
        self._load_audio()
    
    def _validate_files(self):
        """Validate that required files exist."""
        assert os.path.exists(self.json_path), f"Missing JSON at {self.json_path}"
        assert os.path.exists(self.audio_path), f"Missing audio at {self.audio_path}"
    
    def _load_data(self):
        """Load and process JSON data into DataFrame."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        segments = data.get("segments", [])
        if not segments:
            raise ValueError("No segments found in JSON under key 'segments'.")
        
        # Normalize into a DataFrame
        self.df = pd.DataFrame(segments)[["start", "end", "speaker", "text"]].copy()
        self.df["start"] = self.df["start"].astype(float)
        self.df["end"] = self.df["end"].astype(float)
        self.df["duration_s"] = (self.df["end"] - self.df["start"]).round(3)
        self.df.insert(0, "id", range(len(self.df)))  # stable row id
    
    def _load_audio(self):
        """Load the full audio file."""
        self.full_audio, self.sr = sf.read(self.audio_path, always_2d=False)
        is_stereo = self.full_audio.ndim == 2
        n_samples = self.full_audio.shape[0] if not is_stereo else self.full_audio.shape[0]
        self.duration_total = n_samples / self.sr
    
    def get_filtered_table(self, speaker_filter: str, query: str):
        """Return a filtered DataFrame for the UI."""
        dd = self.df.copy()
        if speaker_filter and speaker_filter != "All":
            dd = dd[dd["speaker"] == speaker_filter]
        if query:
            q = query.strip().lower()
            dd = dd[dd["text"].str.lower().str.contains(q, na=False)]
        # Nice ordering
        return dd[["id", "speaker", "start", "end", "duration_s", "text"]].reset_index(drop=True)
    
    def get_audio_slice(self, row_idx, current_table):
        """Get audio slice for a specific row."""
        try:
            if row_idx is None or row_idx < 0 or row_idx >= len(current_table):
                return None, "", ""
            
            seg = current_table.iloc[row_idx]
            seg_id = int(seg["id"])
            
            # Get the original row from the full dataframe
            row = self.df[self.df["id"] == seg_id]
            if row.empty:
                return None, "", ""
            
            row = row.iloc[0]
            start_s, end_s, speaker, text = row["start"], row["end"], row["speaker"], row["text"]
            
            # Create audio slice
            start_idx = int(round(start_s * self.sr))
            end_idx = int(round(end_s * self.sr))
            audio_slice = self.full_audio[start_idx:end_idx]
            
            return (self.sr, audio_slice), f"{speaker} — {start_s:.2f}s → {end_s:.2f}s", text
            
        except Exception as e:
            print(f"Error in audio slice: {e}")
            return None, "", ""
    
    def get_concatenated_audio(self, speaker_filter: str, query: str):
        """Concatenate all segments for the current filter (speaker + query)."""
        table = self.get_filtered_table(speaker_filter, query)
        if table.empty:
            return None, "No segments match.", ""
        
        # Build concatenation with 200 ms silence between segments
        silence_len = int(0.2 * self.sr)
        silence = np.zeros((silence_len, self.full_audio.shape[1])) if self.full_audio.ndim == 2 else np.zeros(silence_len)
        
        pieces = []
        transcript_parts = []
        for _, r in table.iterrows():
            start_s, end_s = float(r["start"]), float(r["end"])
            start_idx = int(round(start_s * self.sr))
            end_idx = int(round(end_s * self.sr))
            pieces.append(self.full_audio[start_idx:end_idx])
            transcript_parts.append(f"[{r['speaker']} {start_s:.2f}–{end_s:.2f}] {r['text']}")
            pieces.append(silence)
        
        if pieces:
            # Drop trailing silence
            pieces = pieces[:-1] if len(pieces) > 1 else pieces
            concat = np.concatenate(pieces, axis=0)
        else:
            return None, "No segments match.", ""
        
        # Return as (sr, np.array) for Gradio Audio
        meta = f"Concatenated {len(table)} segment(s)"
        full_text = "\n\n".join(transcript_parts)
        return (self.sr, concat), meta, full_text
    
    def get_speaker_choices(self):
        """Get list of speaker choices for dropdown."""
        return ["All"] + sorted(self.df["speaker"].dropna().unique().tolist())
    
    def get_stats(self):
        """Get basic stats about the dataset."""
        return {
            "total_segments": len(self.df),
            "duration_total": self.duration_total,
            "sample_rate": self.sr,
            "audio_filename": os.path.basename(self.audio_path)
        } 