# Caspi-1.7B CoreML Conversion & Hex Integration

Hebrew ASR for macOS via Caspi-1.7B (Qwen3-ASR fine-tune) running on Neural Engine through FluidAudio/Hex.

## Background

- **Hex** (github.com/kitlangton/Hex) is a macOS dictation app using FluidAudio for Parakeet TDT and WhisperKit for Whisper models
- **Parakeet TDT v3** does not support Hebrew (25 European languages only, no language hint parameter)
- **Caspi-1.7B** (huggingface.co/OzLabs/Caspi-1.7B) is a Hebrew-optimized fine-tune of Qwen3-ASR-1.7B with ~5% WER on Hebrew benchmarks
- **FluidAudio** already has a `Qwen3AsrManager` with CoreML inference for the 0.6B variant
- **mobius** (github.com/FluidInference/mobius) has the open-source conversion scripts at `models/stt/qwen3-asr-0.6b/coreml/`

## Architecture: Caspi-1.7B vs Qwen3-ASR-0.6B

From the HuggingFace config.json files:

### Audio Encoder

| Parameter | 0.6B | Caspi 1.7B | Notes |
|---|---|---|---|
| d_model | 1024 | 1024 | same |
| encoder_layers | 18 | **24** | 6 more layers |
| encoder_attention_heads | 16 | 16 | same |
| encoder_ffn_dim | 4096 | 4096 | same |
| num_mel_bins | 128 | 128 | same |
| n_window | 50 | 50 | same |
| output_dim | 1024 | **2048** | projects to decoder hidden_size |

### Text Decoder

| Parameter | 0.6B | Caspi 1.7B | Notes |
|---|---|---|---|
| hidden_size | 1024 | **2048** | 2x wider |
| intermediate_size | 2816 | **6144** | MLP inner dim |
| num_hidden_layers | 28 | 28 | same |
| num_attention_heads | 16 | 16 | same |
| num_key_value_heads | 8 | 8 | same (GQA) |
| head_dim | 128 | 128 | same |
| vocab_size | 151,936 | 151,936 | same |
| rope_theta | 1,000,000 | 1,000,000 | same |

**Key insight:** Same layer count, same attention geometry, same KV-cache shape. Only the hidden/intermediate widths change. This makes conversion straightforward.

---

## Phase 1: CoreML Conversion

### Prerequisites

- macOS with Apple Silicon
- Python 3.10+ with `uv` installed
- ~10 GB disk space for model weights + converted artifacts
- Clone the Qwen3-ASR source code (needed by the conversion script)

### Step 1.1: Clone repos and set up environment

```bash
# Clone mobius (conversion scripts)
git clone https://github.com/FluidInference/mobius.git
cd mobius

# Clone Qwen3-ASR source (the conversion script imports from this)
# Must be at ../../../../../../qwen3-asr relative to the convert script,
# or adjust the path in _load_qwen3_asr_modules()
git clone https://github.com/QwenLM/Qwen3-ASR.git qwen3-asr

# Go to conversion directory
cd models/stt/qwen3-asr-0.6b/coreml

# Install Python dependencies
uv sync
```

### Step 1.2: Fork the conversion script for 1.7B

Copy the conversion directory to a new location:

```bash
cd mobius/models/stt
cp -r qwen3-asr-0.6b qwen3-asr-1.7b-caspi
cd qwen3-asr-1.7b-caspi/coreml
```

### Step 1.3: Update convert-qwen3-asr.py

Change `DEFAULT_MODEL_ID`:

```python
# Was:
DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
# Change to:
DEFAULT_MODEL_ID = "OzLabs/Caspi-1.7B"
```

### Step 1.4: Update individual_components.py

Find every hardcoded `hidden_size = 1024` and change to `2048`. Specifically:

**Audio encoder wrapper:**
- The audio encoder output_dim changes from 1024 to 2048
- The mel input shape stays the same (1, 128, 100)

**Decoder layer wrapper:**
- `hidden_size = 2048` (was 1024)
- `num_kv_heads = 8` (unchanged)
- `head_dim = 128` (unchanged)
- KV-cache shapes stay the same: `(1, 8, cache_len, 128)` — no change needed

**LM head wrapper:**
- Input: `(1, 1, 2048)` (was 1024)
- Output: `(1, 1, 151936)` (unchanged)
- IMPORTANT: Keep `compute_precision=ct.precision.FLOAT32` — the float16 overflow bug is even more likely at hidden_size=2048 since RMSNorm computes x^2

**Text embedding wrapper:**
- Embedding dim is now 2048 (output shape changes)

**Decoder stack / prefill wrappers:**
- hidden_size references need updating to 2048
- Prefill seq length may need adjusting (the 0.6B uses seq=512)

### Step 1.5: Update convert_stateful_decoder.py

This is the stateful KV-cache decoder. Update:
- `hidden_size = 2048`
- The state shapes for KV cache remain `(1, 8, cache_dim, 128)` — unchanged since head_dim and num_kv_heads are the same

### Step 1.6: Run the conversion

```bash
# Convert all components
uv run python convert-qwen3-asr.py --output-dir ./build/caspi-1.7b

# Or convert individually to debug issues:
uv run python convert-qwen3-asr.py --components audio_encoder --output-dir ./build/caspi-1.7b
uv run python convert-qwen3-asr.py --components embedding --output-dir ./build/caspi-1.7b
uv run python convert-qwen3-asr.py --components lm_head --output-dir ./build/caspi-1.7b
uv run python convert-qwen3-asr.py --components decoder --output-dir ./build/caspi-1.7b
```

This produces:
- `qwen3_asr_audio_encoder.mlpackage`
- `qwen3_asr_embedding.mlpackage` (or embedding weights as raw binary)
- `qwen3_asr_decoder_prefill.mlpackage`
- `qwen3_asr_decoder_stack.mlpackage` (stateful, with KV-cache)
- `qwen3_asr_lm_head.mlpackage`

### Step 1.7: Compile to .mlmodelc

```bash
# Each .mlpackage needs to be compiled
xcrun coremlcompiler compile qwen3_asr_audio_encoder.mlpackage ./compiled/
xcrun coremlcompiler compile qwen3_asr_decoder_prefill.mlpackage ./compiled/
xcrun coremlcompiler compile qwen3_asr_decoder_stack.mlpackage ./compiled/
xcrun coremlcompiler compile qwen3_asr_lm_head.mlpackage ./compiled/
```

### Step 1.8: Validate against PyTorch

Run the converted models against the original PyTorch model on a few Hebrew audio samples to verify WER is comparable. The conversion script may include validation helpers, or use FluidAudio's benchmark CLI once integrated.

### Step 1.9: Optionally quantize

```bash
# Int8 quantization (~halves size, minimal quality loss)
uv run python convert-qwen3-asr.py --output-dir ./build/caspi-1.7b-int8 --dtype int8
```

Expected sizes:
- f32 variant: ~4-5 GB
- int8 variant: ~2-2.5 GB

### Known conversion pitfalls (from the 0.6B technical report)

1. **Float16 overflow in LM head** — hidden states at ~300 magnitude, 300^2 > float16 max. MUST use `compute_precision=ct.precision.FLOAT32` for decoder and LM head. Even more critical at hidden_size=2048.
2. **Cache-length CoreML bug** — cache lengths 112-126 cause catastrophic errors due to a coremltools bug when head_dim=128. The workaround is already in FluidAudio's Swift code (it pads/avoids those lengths). Since head_dim is still 128, same bug applies.
3. **RoPE layout** — Must be concatenated halves, not interleaved. Already handled in FluidAudio's Qwen3RoPE.swift.

---

## Phase 2: Upload to HuggingFace

### Step 2.1: Create a HuggingFace repo

Create a repo like `your-username/caspi-1.7b-coreml` (or request FluidInference to host it).

### Step 2.2: Upload artifacts

```bash
# Install huggingface CLI if needed
pip install huggingface-hub

# Upload both variants
huggingface-cli upload your-username/caspi-1.7b-coreml ./build/caspi-1.7b --repo-type model
# If you have an int8 variant:
# huggingface-cli upload your-username/caspi-1.7b-coreml ./build/caspi-1.7b-int8 --repo-type model
```

Upload structure should mirror FluidInference/qwen3-asr-0.6b-coreml:
```
f32/
  audioEncoder.mlmodelc/
  decoderStateful.mlmodelc/   (or decoder_prefill + decoder_stack)
  embedding_weights.bin
  vocab.json
int8/
  (same structure)
```

### Step 2.3: Write model card

Include source attribution to OzLabs/Caspi-1.7B and Qwen/Qwen3-ASR-1.7B, license (CC-BY-NC-4.0 from Caspi), and benchmark results.

---

## Phase 3: FluidAudio Integration

### Step 3.1: Add model repo to ModelNames.swift

File: `Sources/FluidAudio/ModelNames.swift`

```swift
public enum Repo: String, CaseIterable {
    // ... existing cases ...
    case qwen3Asr = "FluidInference/qwen3-asr-0.6b-coreml/f32"
    case qwen3AsrInt8 = "FluidInference/qwen3-asr-0.6b-coreml/int8"
    // Add:
    case caspiAsr = "your-username/caspi-1.7b-coreml/f32"
    case caspiAsrInt8 = "your-username/caspi-1.7b-coreml/int8"
}
```

And implement the required computed properties (`name`, `remotePath`, `subPath`, `folderName`) following the existing pattern.

### Step 3.2: Add 1.7B config to Qwen3AsrConfig.swift

File: `Sources/FluidAudio/ASR/Qwen3/Qwen3AsrConfig.swift`

The existing config has 0.6B constants hardcoded. Add a way to switch between 0.6B and 1.7B configs. The key constants to add/override:

```swift
// 1.7B text decoder config
static let hiddenSize_1_7B = 2048          // was 1024
static let intermediateSize_1_7B = 6144    // was 2816
// These stay the same:
// numHiddenLayers = 28
// numAttentionHeads = 16
// numKeyValueHeads = 8
// headDim = 128
// vocabSize = 151936

// 1.7B audio encoder config
static let audioEncoderLayers_1_7B = 24    // was 18
static let audioOutputDim_1_7B = 2048      // was 1024
```

### Step 3.3: Add Hebrew to Qwen3AsrConfig.Language

```swift
public enum Language: String, CaseIterable, Sendable {
    // ... existing cases ...
    case hebrew = "he"  // Add this

    public var englishName: String {
        switch self {
        // ... existing cases ...
        case .hebrew: return "Hebrew"
        }
    }
}
```

### Step 3.4: Update Qwen3AsrModels.swift

Add a new variant enum case and wire up the download path:

```swift
public enum Qwen3AsrVariant: String, CaseIterable, Sendable {
    case f32
    case int8
    case caspiF32      // Add
    case caspiInt8     // Add

    public var repo: Repo {
        switch self {
        case .f32: return .qwen3Asr
        case .int8: return .qwen3AsrInt8
        case .caspiF32: return .caspiAsr
        case .caspiInt8: return .caspiAsrInt8
        }
    }
}
```

### Step 3.5: Update Qwen3AsrManager.swift

The manager needs to handle 2048 hidden_size for the 1.7B model. This affects:
- Embedding weight matrix shape: `(151936, 2048)` instead of `(151936, 1024)`
- Prefill input construction
- Any hardcoded hidden_size references

The KV-cache management stays the same since head_dim and num_kv_heads are unchanged.

### Step 3.6: Update Qwen3RoPE.swift

RoPE parameters are the same (head_dim=128, rope_theta=1M), so no changes needed. The rope_scaling config is identical.

### Step 3.7: Build and test FluidAudio

```bash
cd FluidAudio
swift format --in-place --recursive --configuration .swift-format Sources/ Tests/
swift build
swift test

# Run benchmark on Hebrew audio
swift run -c release fluidaudiocli qwen3-transcribe --model caspi --language he test_hebrew.wav
```

---

## Phase 4: Hex Integration

### Step 4.1: Add Caspi as a model option

File: `HexCore/Sources/HexCore/Models/ParakeetModel.swift`

Either extend the existing model enum or create a new one:

```swift
public enum Qwen3Model: String, CaseIterable, Sendable {
    case caspiHebrew = "caspi-1.7b-coreml"

    public var identifier: String { rawValue }
    public var isEnglishOnly: Bool { false }
}
```

### Step 4.2: Add Caspi to curated model list

File: `Hex/Features/Settings/ModelDownload/ModelDownloadFeature.swift`

Add Caspi to the curated models list with appropriate display name, storage size, and badge:

```swift
CuratedModelInfo(
    displayName: "Caspi 1.7B (Hebrew)",
    internalName: "caspi-1.7b-coreml",
    storageSize: "~2.5 GB",  // int8 size
    isDownloaded: false
)
```

Update `prefersEnglishParakeet` / model selection logic to account for the new model type.

### Step 4.3: Create Qwen3Client.swift (or extend ParakeetClient)

File: `Hex/Clients/Qwen3Client.swift`

New client wrapping FluidAudio's Qwen3AsrManager, similar to ParakeetClient.swift but passing the language parameter:

```swift
#if canImport(FluidAudio)
import FluidAudio

actor Qwen3Client {
    private var asr: Qwen3AsrManager?
    private let logger = HexLog.parakeet  // or add a new log category

    func ensureLoaded(modelName: String, progress: @escaping (Progress) -> Void) async throws {
        let models = try await Qwen3AsrModels.downloadAndLoad(
            variant: .caspiF32,  // or determine from modelName
            progressHandler: { p in progress(p) }
        )
        let manager = Qwen3AsrManager()
        try await manager.loadModels(from: models)
        self.asr = manager
    }

    func transcribe(_ url: URL, language: String?) async throws -> String {
        guard let asr else {
            throw NSError(domain: "Qwen3", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Qwen3 ASR not initialized"])
        }
        let samples = try AudioConverter().resampleAudioFile(url)
        let result = try await asr.transcribe(
            audioSamples: samples,
            language: language  // Pass "he" for Hebrew
        )
        return result
    }
}
#endif
```

### Step 4.4: Wire into TranscriptionClient.swift

File: `Hex/Clients/TranscriptionClient.swift`

Add a `Qwen3Client` instance alongside the existing `ParakeetClient`, and add a new branch in the `transcribe` method:

```swift
private var qwen3: Qwen3Client = Qwen3Client()

func transcribe(url:model:options:progressCallback:) async throws -> String {
    // ... existing code ...

    if isQwen3(model) {
        transcriptionLogger.notice("Transcribing with Qwen3/Caspi model=\(model)")
        try await qwen3.ensureLoaded(modelName: model) { p in
            progressCallback(p)
        }
        let language = options.language  // Pass through from settings
        let text = try await qwen3.transcribe(url, language: language)
        return text
    }

    if isParakeet(model) {
        // ... existing Parakeet code ...
    }

    // ... existing Whisper code ...
}

private func isQwen3(_ model: String) -> Bool {
    model.contains("caspi") || model.contains("qwen3-asr")
}
```

### Step 4.5: Show language picker for Qwen3 models

File: `Hex/Features/Settings/SettingsView.swift`

The current code hides the language picker for Parakeet. Qwen3/Caspi should show it:

```swift
// Was:
if ParakeetModel(rawValue: store.hexSettings.selectedModel) == nil {
    LanguageSectionView(store: store)
}

// Change to:
if ParakeetModel(rawValue: store.hexSettings.selectedModel) == nil
   || Qwen3Model(rawValue: store.hexSettings.selectedModel) != nil {
    LanguageSectionView(store: store)
}
```

Also add Hebrew to the `languages.json` bundle if it's not already there.

### Step 4.6: Update model selection logic

File: `Hex/Features/Settings/ModelDownload/ModelDownloadFeature.swift`

Update `preferredParakeetIdentifier` or add equivalent logic so that when the user selects Hebrew as the output language, the app recommends Caspi:

```swift
var preferredModelIdentifier: String {
    if isHebrewSelected {
        return Qwen3Model.caspiHebrew.identifier
    }
    return preferredParakeetIdentifier
}

private var isHebrewSelected: Bool {
    hexSettings.outputLanguage?.lowercased().hasPrefix("he") == true
}
```

---

## Phase 5: Testing

### Hebrew audio test files

Source Hebrew test audio from:
- ivrit-ai datasets on HuggingFace (crowd-transcribe-v5, crowd-recital)
- Record yourself speaking Hebrew
- Hebrew news clips / podcasts

### Validation checklist

- [ ] CoreML conversion produces all model components without errors
- [ ] Converted model produces valid Hebrew text on test audio (not English/garbage)
- [ ] WER is comparable to Caspi's published benchmarks (~5% on eval-d1)
- [ ] Latency is acceptable for dictation (measure RTF on target hardware)
- [ ] Model download works through FluidAudio's download manager
- [ ] Hex settings UI shows Caspi as a model option
- [ ] Language picker appears and Hebrew is selectable when Caspi is chosen
- [ ] End-to-end: hold hotkey, speak Hebrew, correct Hebrew text is typed

### Performance expectations

Based on the 0.6B CoreML benchmarks and the 2x wider architecture:

| Metric | Qwen3-ASR-0.6B CoreML | Caspi-1.7B CoreML (estimated) |
|---|---|---|
| Weight size (f32) | ~2.5 GB | ~5 GB |
| Weight size (int8) | ~0.7 GB | ~1.5 GB |
| Peak memory | ~400 MB | ~800 MB - 1.2 GB |
| RTF (warm) | ~0.09 | ~0.15-0.25 |
| 3s audio clip | ~0.27s | ~0.5-0.75s |

For comparison, Parakeet TDT is RTF ~0.03 warm. Caspi will be noticeably slower but should still feel responsive for short dictation clips.

---

## Open Questions

1. **Caspi license**: CC-BY-NC-4.0 (non-commercial). Is Hex's use case commercial? If so, you'd need to either:
   - Fine-tune the base Qwen3-ASR-1.7B (Apache-2.0) on the same ivrit-ai datasets yourself
   - Contact OzLabs for a commercial license
   - Use the base Qwen3-ASR-1.7B without Hebrew fine-tuning and hope auto-detect works

2. **FluidAudio upstream**: Should these changes go upstream to FluidAudio? Filing an issue or PR on FluidInference/FluidAudio for 1.7B support would benefit the whole community, and they have expertise with the CoreML conversion pitfalls.

3. **Cache-length bug**: The 0.6B conversion documented a CoreML bug at cache lengths 112-126 with head_dim=128. Since head_dim is still 128 in the 1.7B, the same bug applies. Verify the existing workaround in FluidAudio handles it for the wider model.

4. **Audio encoder change**: The audio encoder has 24 layers (vs 18 in 0.6B) and outputs 2048-dim features. The trace input shape stays the same (1, 128, 100 mel frames), but the model is larger. Verify Neural Engine can handle it in one pass.

---

## Summary

| Phase | Effort | Description |
|---|---|---|
| 1. CoreML conversion | 1-2 days | Fork mobius script, update dimensions, run conversion, validate |
| 2. HuggingFace upload | 30 min | Upload .mlmodelc artifacts |
| 3. FluidAudio integration | 1-2 days | Add 1.7B config, Hebrew language, model variant |
| 4. Hex integration | 1 day | New client, UI wiring, language picker |
| 5. Testing | 1 day | Hebrew audio validation, performance benchmarking |

Total: **~1 week** for someone familiar with the codebase, possibly less if the CoreML conversion goes smoothly on the first attempt.
