import Foundation

/// Known Qwen3-ASR Core ML bundles that Hex supports.
public enum QwenModel: String, CaseIterable, Sendable {
	case caspiHebrew = "caspi-1.7b-coreml"

	/// The identifier used throughout the app (matches the on-disk folder name).
	public var identifier: String { rawValue }

	/// Short capability label for UI copy.
	public var capabilityLabel: String { "Hebrew" }
}
