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
- Automatically truncates the silences ( whether dirty / noisy or not ). <br/>
`While trying to ensure more or less consistent ~100ms spacings ( Some deviations are present and expected. )`
- Respects zero-crossing boundaries.
- Respects breathing ( hopefully.. can't promise much if they're too quiet or way too much noise-like. ).
- Doesn't damage word-tails or inter-phonetic gaps ( unlike gating )
- Truncated areas are automatically replaced by pure silence ( in case of noise-contamination between words or sentences. ).
- No need for user input when it comes to adjusting any params or values. All auto.


ㅤ
<br/>
# ⚠️ㅤ**IMPORTANT** ㅤ⚠️
- Models are in prototype stage. They're trained on limited dataset. Better ones expected in some time.
- For now only CUDA ( nvidia ) or CPU.
- Supported sample rates: 32, 40 and 48khz.<br/>
- Silence / Sub-Silence ( noisy ) spacings below 100ms are ignored / not processed by design.
<br/>
 
✨ to-do list ✨
> - Better pretrained models.
 
💡 Ideas / concepts 💡
> - Currently none. Open to your ideas ~
 
 
### ❗ For contact, please join my discord server ❗
 <br/>
 
## Getting Started:

### INSTALLATION:

Run the installation script:

- Double-click `install.bat`.
 
### INFERENCE:
 
To start inference:
- First put the concatenated sample or samples ( .wav or .flac ) into "infer_input" dir.
- Double-click `run-infer.bat`.
- Results will land in "infer_output" dir.<br/>
`( Concatenated = Simply join up all samples / segments into 1 file )`<br/>`NOTE: ( supports multiple samples AND multiple sr. )`
 
### TRAINING:
- Training of custom pretrains is supported. <br/> Instruction regarding that will be published in future.
 
