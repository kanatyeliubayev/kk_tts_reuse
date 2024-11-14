import sys
import argparse
import torch
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model
from scipy.io.wavfile import write
from pathlib import Path

def synthesize(text, output_file):
    vocoder_checkpoint = "exp/vocoder/checkpoint-400000steps.pkl"  # Update if needed
    config_file = "exp/tts_train_raw_char/config.yaml"
    model_path = "exp/tts_train_raw_char/train.loss.ave_5best.pth"
    
    vocoder = load_model(vocoder_checkpoint).to("cpu").eval()
    text2speech = Text2Speech(config_file, model_path, device="cpu")
    
    with torch.no_grad():
        output = text2speech(text)
        wav = vocoder.inference(output["feat_gen"])
    
    write(output_file, 22050, wav.view(-1).cpu().numpy())
    print(f"Audio saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output audio file")
    args = parser.parse_args()
    
    synthesize(args.text, args.output)
