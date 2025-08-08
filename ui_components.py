import gradio as gr
from data_processor import DataProcessor

class UIComponents:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.demo = None
    
    def on_filter_change(self, speaker_filter, query):
        """Handle filter changes in the UI."""
        table = self.data_processor.get_filtered_table(speaker_filter, query)
        return table
    
    def on_row_select(self, evt: gr.SelectData, current_table):
        """Handle row selection in the data table."""
        try:
            # Get the row index from the event
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            return self.data_processor.get_audio_slice(row_idx, current_table)
        except Exception as e:
            print(f"Error in row select: {e}")
            return None, "", ""
    
    def on_concatenate_click(self, speaker_filter: str, query: str):
        """Handle concatenate button click."""
        return self.data_processor.get_concatenated_audio(speaker_filter, query)
    
    def create_interface(self):
        """Create the Gradio interface."""
        stats = self.data_processor.get_stats()
        
        with gr.Blocks(title="TTS Segment Explorer") as demo:
            gr.Markdown(
                f"# TTS Segment Explorer\n"
                f"- **Audio:** `{stats['audio_filename']}` ({stats['duration_total']:.2f}s @ {stats['sample_rate']} Hz)\n"
                f"- **Segments:** {stats['total_segments']}\n\n"
                "Filter by speaker or search text, click a row to hear that exact slice. "
                "You can also play all matching segments concatenated."
            )
            
            with gr.Row():
                speaker_choices = self.data_processor.get_speaker_choices()
                speaker_dd = gr.Dropdown(speaker_choices, value="All", label="Speaker")
                search_tb = gr.Textbox(value="", label="Search text", placeholder="Type to filter by transcript…")
            
            table = gr.Dataframe(
                headers=["id", "speaker", "start", "end", "duration_s", "text"],
                value=self.data_processor.get_filtered_table("All", ""),
                row_count=(stats['total_segments'], "dynamic"),
                wrap=True,
                interactive=False,
                label="Segments (click a row to play)"
            )
            
            with gr.Row():
                seg_audio = gr.Audio(label="Selected Segment", interactive=False)
                seg_meta = gr.Markdown()
            seg_text = gr.Textbox(label="Segment Text", interactive=False, lines=5)
            
            with gr.Accordion("Play all matching segments (by current filter)", open=False):
                all_audio = gr.Audio(label="Concatenated Audio", interactive=False)
                all_meta = gr.Markdown()
                all_text = gr.Textbox(label="Combined Transcript", interactive=False, lines=10)
                play_all_btn = gr.Button("▶️ Concatenate & Play")
            
            # Events
            speaker_dd.change(
                self.on_filter_change, 
                inputs=[speaker_dd, search_tb], 
                outputs=table
            )
            search_tb.change(
                self.on_filter_change, 
                inputs=[speaker_dd, search_tb], 
                outputs=table
            )
            
            # Row selection -> play exact slice
            table.select(
                self.on_row_select, 
                inputs=[table], 
                outputs=[seg_audio, seg_meta, seg_text]
            )
            
            # Button -> play all filtered
            play_all_btn.click(
                self.on_concatenate_click, 
                inputs=[speaker_dd, search_tb], 
                outputs=[all_audio, all_meta, all_text]
            )
        
        self.demo = demo
        return demo 