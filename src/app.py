import gradio as gr
import torch
import numpy as np
import os
import io
import base64
from kokoro import KModel, KPipeline

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()

# Initialize the model
model = KModel().to('cuda' if CUDA_AVAILABLE else 'cpu').eval()

# Initialize pipelines for different language codes (using 'a' for English)
pipelines = {'a': KPipeline(lang_code='a', model=False)}

# Custom pronunciation for "kokoro"
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'

def text_to_audio(text, speed=1.0):
    """Convert text to audio using Kokoro model.
    
    Args:
        text: The text to convert to speech
        speed: Speech speed multiplier (0.5-2.0, where 1.0 is normal speed)
        
    Returns:
        Audio data as a tuple of (sample_rate, audio_array)
    """
    if not text:
        return None
    
    pipeline = pipelines['a']  # Use English pipeline
    voice = "af_heart"  # Default voice (US English, female, Heart)
    
    # Process the text
    pack = pipeline.load_voice(voice)
    
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        
        # Generate audio
        try:
            audio = model(ps, ref_s, speed)
        except Exception as e:
            raise gr.Error(f"Error generating audio: {str(e)}")
        
        # Return the audio with 24kHz sample rate
        return 24000, audio.numpy()
    
    return None

def text_to_audio_b64(text, speed=1.0):
    """Convert text to audio and return as base64 encoded WAV file.
    
    Args:
        text: The text to convert to speech
        speed: Speech speed multiplier (0.5-2.0, where 1.0 is normal speed)
        
    Returns:
        Base64 encoded WAV file as a string
    """
    import soundfile as sf
    
    result = text_to_audio(text, speed)
    if result is None:
        return None
    
    sample_rate, audio_data = result
    
    # Save to BytesIO object
    wav_io = io.BytesIO()
    sf.write(wav_io, audio_data, sample_rate, format='WAV')
    wav_io.seek(0)
    
    # Convert to base64
    wav_b64 = base64.b64encode(wav_io.read()).decode('utf-8')
    return wav_b64

# Create Gradio interface
with gr.Blocks(title="Kokoro Text-to-Audio MCP") as app:
    gr.Markdown("# 🎵 Kokoro Text-to-Audio MCP")
    gr.Markdown("Convert text to speech using the Kokoro-82M model")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter your text",
                placeholder="Type something to convert to audio...",
                lines=5
            )
            speed_slider = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Speech Speed"
            )
            submit_btn = gr.Button("Generate Audio")
        
        with gr.Column():
            audio_output = gr.Audio(label="Generated Audio", type="numpy")
    
    submit_btn.click(
        fn=text_to_audio,
        inputs=[text_input, speed_slider],
        outputs=[audio_output]
    )
    
    gr.Markdown("### Usage Tips")
    gr.Markdown("- Adjust the speed slider to modify the pace of speech")
    
    # Add section about MCP support
    with gr.Accordion("MCP Support (for LLMs)", open=False):
        gr.Markdown("""
        ### MCP Support
        
        This app supports the Model Context Protocol (MCP), allowing Large Language Models like Claude Desktop to use it as a tool.
        
        To use this app with an MCP client, add the following configuration:
        
        ```json
        {
          "mcpServers": {
            "kokoroTTS": {
              "url": "https://fdaudens-kokoro-mcp.hf.space/gradio_api/mcp/sse"
            }
          }
        }
        ```
        
        Replace `your-app-url.hf.space` with your actual Hugging Face Space URL.
        """)

# Launch the app with MCP support
if __name__ == "__main__":
    # Check for environment variable to enable MCP
    enable_mcp = os.environ.get('GRADIO_MCP_SERVER', 'False').lower() in ('true', '1', 't')
    
    app.launch(mcp_server=True)