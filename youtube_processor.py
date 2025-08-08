import subprocess
import sys
import argparse

def process_youtube_video(url, assemblyai_key):
    """Process a YouTube video with AssemblyAI."""
    try:
        # Run the TTS dataset maker script
        cmd = [
            sys.executable, 
            "tts_dataset_maker.py", 
            url, 
            "--assemblyai-key", 
            assemblyai_key
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("YouTube video processed successfully!")
            print("Output files should be in the 'output' directory.")
        else:
            print(f"Error processing video: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error running TTS dataset maker: {e}")
        return False
    
    return True

def main():
    """Main function for YouTube processing."""
    parser = argparse.ArgumentParser(description="Process YouTube video for TTS dataset")
    parser.add_argument("url", help="YouTube URL to process")
    parser.add_argument("--assemblyai-key", required=True, help="AssemblyAI API key")
    
    args = parser.parse_args()
    
    print(f"Processing YouTube video: {args.url}")
    if process_youtube_video(args.url, args.assemblyai_key):
        print("Processing completed successfully!")
    else:
        print("Processing failed.")

if __name__ == "__main__":
    main() 