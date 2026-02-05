# <p align="center">` SmartCutter ` </p>

<p align="center"> ㅤㅤ👇 You can join my discord server below ( RVC / AI Audio friendly ) 👇ㅤㅤ </p>

</p>
<p align="center">
  <a href="https://discord.gg/nQFpNBvvd3" target="_blank"> Codename's Sanctuary</a>
</p>

<p align="center"> ㅤㅤ👆 To stay up-to-date with advancements, hang out or get support 👆ㅤㅤ </p>

## <p align="center"> A lil bit more about the project:

### <p align="center"> Machine Learning based silence-truncation. <br/> Made with Applio / RVC and my Codename-RVC-Fork-4 in mind. ✨ <br/> <br/>
### Features:
- Automatically truncates the silences ( whether dirty / noisy or not.. tho there's limits. It's not a noise-gate trimmer afterall haha. ). <br/>
`While trying to ensure more or less consistent ~100ms spacings ( Some deviations are present and expected. )`
- Respects zero-crossing boundaries.
- Respects breathing ( hopefully.. can't promise much if they're too quiet or way too much noise-like. ).
- Doesn't damage word-tails or inter-phonetic gaps ( unlike gating )
- Truncated areas are automatically replaced by pure silence ( in case of noise-contamination between words or sentences. ).
- No need for user input when it comes to adjusting any params or values. All's handled automatically.

## Scenarios it handles very reliably:
<img width="1621" height="755" alt="image" src="https://github.com/user-attachments/assets/5da19052-61db-47e5-88e7-90bf76311c60" /> <br/>
## Scenarios it might fail or the reliability is uncertain:
### 1. <img width="1621" height="751" alt="image" src="https://github.com/user-attachments/assets/1fa04d01-57cf-43c0-b3f1-1bbf819dbbaf" /> <br>
### 2. <img width="1630" height="759" alt="image" src="https://github.com/user-attachments/assets/5638f038-62c2-4904-8268-0de2caeb941f" />
### Therefore, for such " hard cases " ( 1, 2 ) spectral de-noise ( or gating if you're careful ) is recommended.
ㅤ
<br/>
# ⚠️ㅤ**IMPORTANT** ㅤ⚠️
- For now only CUDA ( nvidia ) or CPU.
- Supported sample rates: 32, 40 and 48khz.<br/>
- Silence / Sub-Silence ( noisy ) spacings below 100ms are ignored / not processed by design.
- There are limits, it is still a very-low-noise or pure silence focused truncator.
( So keep in mind models might hiccup on some really hard cases. ) <br/>
<br/>
 
✨ to-do list ✨
> - ~~Better pretrained models.~~ done.
> - Eventually ( potentially ) move over to v5 arch ~ Bigger dataset's needed.
 
💡 Ideas / concepts 💡
> - Currently none. Open to your ideas ~
 
 
### ❗ For contact, please join my discord server ❗
 <br/>
 
## Getting Started:

### INSTALLATION:

Run the installation script:

- Double-click `install.bat`.
 
### PRETRAINED MODELS:

- Download all 3 checkpoints ( Each ~54mb ):)<br/>
[v3_model_48000](https://huggingface.co/Codename0/SmartCutter/resolve/main/v3_model_48000.pth?download=true)<br/>
[v3_model_40000](https://huggingface.co/Codename0/SmartCutter/resolve/main/v3_model_40000.pth?download=true)<br/>
[v3_model_32000](https://huggingface.co/Codename0/SmartCutter/resolve/main/v3_model_32000.pth?download=true)<br/>
- Put them in SmartCutter's "ckpts" folder
 
### INFERENCE:
 
To start inference:
- First put the concatenated sample or samples ( .wav or .flac ) into "infer_input" dir.
- Double-click `run-infer.bat`.
- Results will land in "infer_output" dir.<br/>
`( Concatenated = Simply join up all samples / segments into 1 file )`<br/><br/>`NOTE: supports multiple samples AND multiple sr.`
 
### TRAINING:
- Training of custom pretrains is supported. <br/> Instruction regarding that will be published in future.
 
