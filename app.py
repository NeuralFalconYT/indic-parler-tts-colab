# Copied From here https://huggingface.co/spaces/ai4bharat/indic-parler-tts/tree/main
import io
import os
import math
from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np
# import spaces
import gradio as gr
import torch
import nltk


from parler_tts import ParlerTTSForConditionalGeneration
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed

nltk.download('punkt_tab')

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32

repo_id = "ai4bharat/indic-parler-tts-pretrained"
finetuned_repo_id = "ai4bharat/indic-parler-tts"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    repo_id, attn_implementation="eager", torch_dtype=torch_dtype,
).to(device)
finetuned_model = ParlerTTSForConditionalGeneration.from_pretrained(
    finetuned_repo_id, attn_implementation="eager", torch_dtype=torch_dtype,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(repo_id)
description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

SAMPLE_RATE = feature_extractor.sampling_rate
SEED = 42

default_text = "Please surprise me and speak in whatever voice you enjoy."
examples = [
    [
        "मुले बागेत खेळत आहेत आणि पक्षी किलबिलाट करत आहेत.",
        "Sunita speaks slowly in a calm, moderate-pitched voice, delivering the news with a neutral tone. The recording is very high quality with no background noise.",
        3.0
    ],
    [
        "ಉದ್ಯಾನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟವಾಡುತ್ತಿದ್ದಾರೆ ಮತ್ತು ಪಕ್ಷಿಗಳು ಚಿಲಿಪಿಲಿ ಮಾಡುತ್ತಿವೆ.",
        "Suresh speaks slowly in a low-pitched, calm voice, with a neutral tone, perfect for narration. The recording is very high quality with no background noise.",
        3.0
    ],
    [
        "বাচ্চারা বাগানে খেলছে আর পাখি কিচিরমিচির করছে।",
        "Aditi speaks at a moderate pace and pitch, with a clear, neutral tone and no emotional emphasis. The recording is very high quality with no background noise.",
        3.0
    ],
    [
        "పిల్లలు తోటలో ఆడుకుంటున్నారు, పక్షుల కిలకిలరావాలు.",
        "Prakash speaks slowly in a low-pitched, calm voice, with a neutral tone, perfect for narration. The recording is very high quality with no background noise.",
        3.0
    ],
    [
        "పిల్లలు తోటలో ఆడుకుంటున్నారు, పక్షుల కిలకిలరావాలు.",
        "Prakash speaks slowly in a low-pitched, calm voice, with a neutral tone, perfect for narration. The recording is very high quality with no background noise.",
        3.0
    ],
    [
        "This is the best time of my life, Bartley,' she said happily",
        "A male speaker with a low-pitched voice speaks with a British accent at a fast pace in a small, confined space with very clear audio and an animated tone.",
        3.0
    ],
    [
        "Montrose also, after having experienced still more variety of good and bad fortune, threw down his arms, and retired out of the kingdom.",
        "A female speaker with a slightly low-pitched, quite monotone voice speaks with an American accent at a slightly faster-than-average pace in a confined space with very clear audio.",
        3.0
    ],
    [
        "बगीचे में बच्चे खेल रहे हैं और पक्षी चहचहा रहे हैं।",
        "Rohit speaks with a slightly high-pitched voice delivering his words at a slightly slow pace in a small, confined space with a touch of background noise and a quite monotone tone.",
        3.0
    ],
    [
        "കുട്ടികൾ പൂന്തോട്ടത്തിൽ കളിക്കുന്നു, പക്ഷികൾ ചിലയ്ക്കുന്നു.",
        "Anjali speaks with a low-pitched voice delivering her words at a fast pace and an animated tone, in a very spacious environment, accompanied by noticeable background noise.",
        3.0
    ],
    [
        "குழந்தைகள் தோட்டத்தில் விளையாடுகிறார்கள், பறவைகள் கிண்டல் செய்கின்றன.",
        "Jaya speaks with a slightly low-pitched, quite monotone voice at a slightly faster-than-average pace in a confined space with very clear audio.",
        3.0
    ]
]


finetuned_examples = [
    [
        "मुले बागेत खेळत आहेत आणि पक्षी किलबिलाट करत आहेत.",
        "Sunita speaks slowly in a calm, moderate-pitched voice, delivering the news with a neutral tone. The recording is very high quality with no background noise.",
        3.0
    ],
    [
        "ಉದ್ಯಾನದಲ್ಲಿ ಮಕ್ಕಳ ಆಟವಾಡುತ್ತಿದ್ದಾರೆ ಮತ್ತು ಪಕ್ಷಿಗಳು ಚಿಲಿಪಿಲಿ ಮಾಡುತ್ತಿವೆ.",
        "Suresh speaks slowly in a low-pitched, calm voice, with a neutral tone, perfect for narration. The recording is very high quality with no background noise.",
        3.0
    ],
    [
        "বাচ্চারা বাগানে খেলছে আর পাখি কিচিরমিচির করছে।",
        "Aditi speaks at a moderate pace and pitch, with a clear, neutral tone and no emotional emphasis. The recording is very high quality with no background noise.",
        3.0
    ],
    [
        "పిల్లలు తోటలో ఆడుకుంటున్నారు, పక్షుల కిలకిలరావాలు.",
        "Prakash speaks slowly in a low-pitched, calm voice, with a neutral tone, perfect for narration. The recording is very high quality with no background noise.",
        3.0
    ],
    [
        "పిల్లలు తోటలో ఆడుకుంటున్నారు, పక్షుల కిలకిలరావాలు.",
        "Prakash speaks slowly in a low-pitched, calm voice, with a neutral tone, perfect for narration. The recording is very high quality with no background noise.",
        3.0
    ],
    [
        "This is the best time of my life, Bartley,' she said happily",
        "A male speaker with a low-pitched voice speaks with a British accent at a fast pace in a small, confined space with very clear audio and an animated tone.",
        3.0
    ],
    [
        "Montrose also, after having experienced still more variety of good and bad fortune, threw down his arms, and retired out of the kingdom.",
        "A female speaker with a slightly low-pitched, quite monotone voice speaks with an American accent at a slightly faster-than-average pace in a confined space with very clear audio.",
        3.0
    ],
    [
        "बगीचे में बच्चे खेल रहे हैं और पक्षी चहचहा रहे हैं।",
        "Rohit speaks with a slightly high-pitched voice delivering his words at a slightly slow pace in a small, confined space with a touch of background noise and a quite monotone tone.",
        3.0
    ],
    [
        "കുട്ടികൾ പൂന്തോട്ടത്തിൽ കളിക്കുന്നു, പക്ഷികൾ ചിലയ്ക്കുന്നു.",
        "Anjali speaks with a low-pitched voice delivering her words at a fast pace and an animated tone, in a very spacious environment, accompanied by noticeable background noise.",
        3.0
    ],
    [
        "குழந்தைகள் தோட்டத்தில் விளையாடுகிறார்கள், பறவைகள் கிண்டல் செய்கின்றன.",
        "Jaya speaks with a slightly low-pitched, quite monotone voice at a slightly faster-than-average pace in a confined space with very clear audio.",
        3.0
    ]
]


def numpy_to_mp3(audio_array, sampling_rate):
    # Normalize audio_array if it's floating-point
    if np.issubdtype(audio_array.dtype, np.floating):
        max_val = np.max(np.abs(audio_array))
        audio_array = (audio_array / max_val) * 32767  # Normalize to 16-bit range
        audio_array = audio_array.astype(np.int16)

    # Create an audio segment from the numpy array
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sampling_rate,
        sample_width=audio_array.dtype.itemsize,
        channels=1
    )

    # Export the audio segment to MP3 bytes - use a high bitrate to maximise quality
    mp3_io = io.BytesIO()
    audio_segment.export(mp3_io, format="mp3", bitrate="320k")

    # Get the MP3 bytes
    mp3_bytes = mp3_io.getvalue()
    mp3_io.close()

    return mp3_bytes

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

# @spaces.GPU
def generate_base(text, description,):
    # Initialize variables
    chunk_size = 25  # Process max 25 words or a sentence at a time
    
    # Tokenize the full text and description
    inputs = description_tokenizer(description, return_tensors="pt").to(device)

    sentences_text = nltk.sent_tokenize(text) # this gives us a list of sentences
    curr_sentence = ""
    chunks = []
    for sentence in sentences_text:
        candidate = " ".join([curr_sentence, sentence])
        if len(candidate.split()) >= chunk_size:
            chunks.append(curr_sentence)
            curr_sentence = sentence
        else:
            curr_sentence = candidate

    if curr_sentence != "":
        chunks.append(curr_sentence)
        
    print(chunks)

    all_audio = []
    
    # Process each chunk
    for chunk in chunks:
        # Tokenize the chunk
        prompt = tokenizer(chunk, return_tensors="pt").to(device)
        
        # Generate audio for the chunk
        generation = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
            do_sample=True,
            return_dict_in_generate=True
        )
            
        # Extract audio from generation
        if hasattr(generation, 'sequences') and hasattr(generation, 'audios_length'):
            audio = generation.sequences[0, :generation.audios_length[0]]
            audio_np = audio.to(torch.float32).cpu().numpy().squeeze()
            if len(audio_np.shape) > 1:
                audio_np = audio_np.flatten()
            all_audio.append(audio_np)
    
    # Combine all audio chunks
    combined_audio = np.concatenate(all_audio)
    
    # Convert to expected format and yield
    print(f"Sample of length: {round(combined_audio.shape[0] / sampling_rate, 2)} seconds")
    yield numpy_to_mp3(combined_audio, sampling_rate=sampling_rate)


# @spaces.GPU
def generate_finetuned(text, description):
    # Initialize variables
    chunk_size = 25  # Process max 25 words or a sentence at a time
    
    # Tokenize the full text and description
    inputs = description_tokenizer(description, return_tensors="pt").to(device)

    sentences_text = nltk.sent_tokenize(text) # this gives us a list of sentences
    curr_sentence = ""
    chunks = []
    for sentence in sentences_text:
        candidate = " ".join([curr_sentence, sentence])
        if len(candidate.split()) >= chunk_size:
            chunks.append(curr_sentence)
            curr_sentence = sentence
        else:
            curr_sentence = candidate

    if curr_sentence != "":
        chunks.append(curr_sentence)
        
    print(chunks)
    
    all_audio = []
    
    # Process each chunk
    for chunk in chunks:
        # Tokenize the chunk
        prompt = tokenizer(chunk, return_tensors="pt").to(device)

        # Generate audio for the chunk
        generation = finetuned_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
            do_sample=True,
            return_dict_in_generate=True
        )

        # Extract audio from generation
        if hasattr(generation, 'sequences') and hasattr(generation, 'audios_length'):
            audio = generation.sequences[0, :generation.audios_length[0]]
            audio_np = audio.to(torch.float32).cpu().numpy().squeeze()
            if len(audio_np.shape) > 1:
                audio_np = audio_np.flatten()
            all_audio.append(audio_np)
    
    # Combine all audio chunks
    combined_audio = np.concatenate(all_audio)
    
    # Convert to expected format and yield
    print(f"Sample of length: {round(combined_audio.shape[0] / sampling_rate, 2)} seconds")
    yield numpy_to_mp3(combined_audio, sampling_rate=sampling_rate)


css = """
        #share-btn-container {
            display: flex;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            background-color: #000000;
            justify-content: center;
            align-items: center;
            border-radius: 9999px !important; 
            width: 13rem;
            margin-top: 10px;
            margin-left: auto;
            flex: unset !important;
        }
        #share-btn {
            all: initial;
            color: #ffffff;
            font-weight: 600;
            cursor: pointer;
            font-family: 'IBM Plex Sans', sans-serif;
            margin-left: 0.5rem !important;
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
            right:0;
        }
        #share-btn * {
            all: unset !important;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
"""
with gr.Blocks(css=css) as block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 700px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                  Parler-TTS 🗣️
                </h1>
              </div>
            </div>
        """
    )
    gr.HTML(
        f"""
        <p><a href="https://github.com/huggingface/Parler-TTS">ParlerTTS</a> is a training and inference library for high-quality text-to-speech (TTS) models. This demonstration highlights the flexibility of the IndicParlerTTS model, which generates natural, expressive speech for over 22 Indian languages, using a simple text prompt to control features like speaker style, tone, pitch, pace, and more.</p>

        <p>Tips for effective usage:
        <ul>
            <li>Use detailed captions to describe the speaker and desired characteristics (e.g., "Aditi speaks in a slightly expressive tone, with clear audio quality and a moderate pace.").</li>
            <li>For best results, reference specific named speakers provided in the model card on the <a href="https://huggingface.co/ai4bharat/indic-parler-tts#%F0%9F%8E%AF-using-a-specific-speaker">model page</a>.</li>
            <li>Include terms like <b>"very clear audio"</b> or <b>"slightly noisy audio"</b> to control the audio quality and background ambiance.</li>
            <li>Punctuation can be used to shape prosody (e.g., commas add pauses for natural phrasing).</li>
            <li>If unsure about what caption to use, you can start with: <b>"The speaker speaks naturally. The recording is very high quality with no background noise."</b></li>
        </ul>
        </p>
        """
    )

    with gr.Tab("Finetuned"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Input Text", lines=2, value=finetuned_examples[0][0], elem_id="input_text")
                description = gr.Textbox(label="Description", lines=2, value=finetuned_examples[0][1], elem_id="input_description")
                run_button = gr.Button("Generate Audio", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Parler-TTS generation", format="mp3", elem_id="audio_out", autoplay=True)

        inputs = [input_text, description]
        outputs = [audio_out]
        gr.Examples(examples=finetuned_examples, fn=generate_finetuned, inputs=inputs, outputs=outputs, cache_examples=False)
        run_button.click(fn=generate_finetuned, inputs=inputs, outputs=outputs, queue=True)

    with gr.Tab("Pretrained"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Input Text", lines=2, value=default_text, elem_id="input_text")
                description = gr.Textbox(label="Description", lines=2, value="", elem_id="input_description")
                run_button = gr.Button("Generate Audio", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Parler-TTS generation", format="mp3", elem_id="audio_out", autoplay=True)

        inputs = [input_text, description]
        outputs = [audio_out]
        gr.Examples(examples=examples, fn=generate_base, inputs=inputs, outputs=outputs, cache_examples=False)
        run_button.click(fn=generate_base, inputs=inputs, outputs=outputs, queue=True)


    gr.HTML(
        """
        If you'd like to learn more about how the model was trained or explore fine-tuning it yourself, visit the <a href="https://github.com/huggingface/parler-tts">Parler-TTS</a> repository on GitHub. The Parler-TTS codebase and associated checkpoints are licensed under the <a href="https://github.com/huggingface/parler-tts/blob/main/LICENSE">Apache 2.0 license</a>.</p>
                """
    )

# block.queue()
# block.launch(share=True)
import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
  block.queue().launch(debug=debug, share=share)
if __name__ == "__main__":
    main()  
