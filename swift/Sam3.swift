import UIKit
import CoreML
import Accelerate // For faster array math if needed, though standard loops used here for clarity
class SAM3Predictor {
  // MARK: - Configuration
  private let width = 1008
  private let height = 1008
  private let contextLength = 32 // Standard CLIP context length
  // MARK: - Models
  private var encoder: MLModel?
  private var decoder: MLModel?
  private var textEncoder: MLModel?
  private var tokenizer: SimpleTokenizer?
  // MARK: - Initialization
  /// Initializes the predictor by loading Core ML models and the BPE tokenizer.
  /// - Parameters:
  ///  - encoderName: Name of the encoder resource (e.g. "SAM3_Encoder")
  ///  - decoderName: Name of the decoder resource (e.g. "SAM3_Decoder")
  ///  - textEncoderName: Name of the text encoder resource (e.g. "SAM3_TextEncoder")
  ///  - bpeFileName: Name of the BPE vocab file (e.g. "bpe_simple_vocab_16e6")
  init(encoderName: String = "SAM3_Encoder",
     decoderName: String = "SAM3_Decoder",
     textEncoderName: String = "SAM3_TextEncoder",
     bpeFileName: String = "bpe_simple_vocab_16e6") {
    // 1. Load Core ML Models
    self.encoder = loadModel(named: encoderName, computeUnits: .cpuOnly)
    self.decoder = loadModel(named: decoderName, computeUnits: .cpuOnly)
    self.textEncoder = loadModel(named: textEncoderName, computeUnits: .cpuOnly)
    // 2. Initialize Tokenizer (Updated to use URL)
    // Note: bpeFileName should not include the extension here if it's passed in 'withExtension'
    let cleanName = bpeFileName.replacingOccurrences(of: ".txt", with: "")
    if let bpeURL = Bundle.main.url(forResource: cleanName, withExtension: "txt"),
      let bpeContent = try? String(contentsOf: bpeURL, encoding: .utf8) {
      self.tokenizer = SimpleTokenizer(bpeContent: bpeContent, contextLength: contextLength)
      print(":marca_de_verificación_blanca: Tokenizer loaded from URL")
    } else {
      print(":advertencia: Failed to load Tokenizer BPE file. Checked: \(cleanName).txt")
    }
  }
  func unloadModels() {
    encoder = nil
    decoder = nil
    textEncoder = nil
    // tokenizer is small, but we can nil it too for completeness
    tokenizer = nil
    print(":papelera: Core ML models unloaded from memory.")
  }
  private func loadModel(named name: String, computeUnits: MLComputeUnits) -> MLModel? {
    guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") ??
            Bundle.main.url(forResource: name, withExtension: "mlpackage") else {
      print(":advertencia: Could not find model: \(name)")
      print("Bundle.main.resourcePath: \(Bundle.main.resourcePath ?? "nil")")
      return nil
    }
    do {
      let config = MLModelConfiguration()
      config.computeUnits = computeUnits
      return try MLModel(contentsOf: url, configuration: config)
    } catch {
      print(":x: Error loading \(name): \(error)")
      return nil
    }
  }
  // MARK: - Main Prediction Function
  /// Runs the full SAM3 inference pipeline.
  /// - Parameters:
  ///  - image: Input UIImage (will be resized to 1008x1008)
  ///  - prompt: Text prompt (e.g., "eyebrow")
  ///  - iouThreshold: Minimum score to accept a mask (default 0.6)
  /// - Returns: A binary UIImage mask (white = object, black = background), or nil if failed.
  func predict(image: UIImage, prompt: String, iouThreshold: Float = 0.6) -> (image: UIImage, iou: Float)? {
    guard let encoder = encoder, let decoder = decoder, let textEncoder = textEncoder, let tokenizer = tokenizer else {
      print(":x: Models not fully initialized.")
      return nil
    }
    // 1. Preprocess Image
    guard let resizedBuffer = image.pixelBuffer(width: width, height: height) else {
      print(":x: Failed to process image.")
      return nil
    }
    do {
      // 2. Run Image Encoder
      let encoderInput = try MLDictionaryFeatureProvider(dictionary: ["input_image": resizedBuffer])
      let encoderOut = try encoder.prediction(from: encoderInput)
      // 3. Run Text Encoder
      let (langFeats, langMask) = try runTextEncoder(prompt: prompt, tokenizer: tokenizer, model: textEncoder)
      // 4. Prepare Decoder Inputs
      var decoderInputs: [String: Any] = [
        "fpn0": encoderOut.featureValue(for: "fpn0")!.multiArrayValue!,
        "fpn1": encoderOut.featureValue(for: "fpn1")!.multiArrayValue!,
        "fpn2": encoderOut.featureValue(for: "fpn2")!.multiArrayValue!,
        "pos0": encoderOut.featureValue(for: "pos0")!.multiArrayValue!,
        "pos1": encoderOut.featureValue(for: "pos1")!.multiArrayValue!,
        "pos2": encoderOut.featureValue(for: "pos2")!.multiArrayValue!,
        "lang_feats": langFeats,
        "lang_mask": langMask
      ]
      addDummyGeometry(to: &decoderInputs)
      // 5. Run Mask Decoder
      let decoderInputProvider = try MLDictionaryFeatureProvider(dictionary: decoderInputs)
      let decoderOut = try decoder.prediction(from: decoderInputProvider)
      // 6. Post-Processing
      guard let rawMasks = decoderOut.featureValue(for: "masks")?.multiArrayValue,
         let rawIOU = decoderOut.featureValue(for: "iou_scores")?.multiArrayValue else {
        return nil
      }
      return processOutput(masks: rawMasks, iouScores: rawIOU, threshold: iouThreshold)
    } catch {
      print(":x: Prediction failed: \(error)")
      return nil
    }
  }
  // MARK: - Helpers
  private func runTextEncoder(prompt: String, tokenizer: SimpleTokenizer, model: MLModel) throws -> (MLMultiArray, MLMultiArray) {
    // 1. Tokenize
    // Returns [[Int]], we take the first (and only) batch item
    let tokenIds = tokenizer.tokenize(prompt)[0]
    // 2. Create Input MultiArray [1, 77]
    let shape = [1, NSNumber(value: contextLength)]
    let inputIdsArray = try MLMultiArray(shape: shape, dataType: .int32)
    for (i, id) in tokenIds.enumerated() {
      inputIdsArray[i] = NSNumber(value: id)
    }
    // 3. Predict
    let input = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIdsArray])
    let output = try model.prediction(from: input)
    let feats = output.featureValue(for: "lang_feats")!.multiArrayValue!
    let mask = output.featureValue(for: "lang_mask")!.multiArrayValue!
    return (feats, mask)
  }
  private func addDummyGeometry(to inputs: inout [String: Any]) {
    // Create dummy points and boxes filled with zeros (or specific padding values)
    // Shapes based on Python script: Points [1, 5, 2], Labels [1, 5], Boxes [1, 5, 4]
    try? inputs["point_coords"] = MLMultiArray(shape: [1, 5, 2], dataType: .float32)
    let pointLabels = try! MLMultiArray(shape: [1, 5], dataType: .int32)
    // Fill labels with 1 (or -1 depending on model specifics, Python used 1)
    for i in 0..<pointLabels.count { pointLabels[i] = 1 }
    inputs["point_labels"] = pointLabels
    try? inputs["box_coords"] = MLMultiArray(shape: [1, 5, 4], dataType: .float32)
    try? inputs["box_labels"] = MLMultiArray(shape: [1, 5], dataType: .int32)
  }
  private func processOutput(masks: MLMultiArray, iouScores: MLMultiArray, threshold: Float) -> (image: UIImage, iou: Float)? {
    let count = iouScores.count
    let maskHeight = masks.shape[2].intValue
    let maskWidth = masks.shape[3].intValue
    let totalPixels = maskHeight * maskWidth
    // 1. Sigmoid IOU and filter
    var bestIndices: [Int] = []
    var maxIOU: Float = -Float.greatestFiniteMagnitude
    var bestSingleIndex = 0
    for i in 0..<count {
      let score = iouScores[i].floatValue
      // Sigmoid function: 1 / (1 + e^-x)
      let sigmoidScore = 1.0 / (1.0 + exp(-score))
      if sigmoidScore > maxIOU {
        maxIOU = sigmoidScore
        bestSingleIndex = i
      }
      if sigmoidScore > threshold {
        bestIndices.append(i)
      }
    }
    // Fallback if no mask meets threshold
    if bestIndices.isEmpty {
      bestIndices.append(bestSingleIndex)
      // For debugging, print the 10 best indices
      let sortedIOUIndices = (0..<count).sorted { iouScores[$0].floatValue > iouScores[$1].floatValue }
      let top10Indices = sortedIOUIndices.prefix(10)
      print("No mask met threshold. Falling back to best mask. Top 10 IOU scores:")
      for index in top10Indices {
        let score = iouScores[index].floatValue
        let sigmoidScore = 1.0 / (1.0 + exp(-score))
        print(" Index \(index): Raw IOU = \(score), Sigmoid IOU = \(sigmoidScore)")
      }
    }
    // 2. Combine Masks
    var pixelData = [UInt8](repeating: 0, count: totalPixels)
    masks.withUnsafeBufferPointer(ofType: Float.self) { ptr in
      for maskIdx in bestIndices {
        let offset = maskIdx * totalPixels
        for p in 0..<totalPixels {
          if pixelData[p] == 0 {
            let val = ptr[offset + p]
            if val > 0.0 {
              pixelData[p] = 255
            }
          }
        }
      }
    }
    if let resultImage = imageFromGrayscaleBuffer(data: pixelData, width: maskWidth, height: maskHeight) {
      return (resultImage, maxIOU)
    }
    return nil
  }
  private func imageFromGrayscaleBuffer(data: [UInt8], width: Int, height: Int) -> UIImage? {
    guard data.count == width * height else { return nil }
    let colorSpace = CGColorSpaceCreateDeviceGray()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
    guard let provider = CGDataProvider(data: Data(data) as CFData),
       let cgImage = CGImage(width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bitsPerPixel: 8,
                  bytesPerRow: width,
                  space: colorSpace,
                  bitmapInfo: bitmapInfo,
                  provider: provider,
                  decode: nil,
                  shouldInterpolate: false,
                  intent: .defaultIntent) else {
      return nil
    }
    return UIImage(cgImage: cgImage)
  }
}
// MARK: - UIImage Extension for Resizing
extension UIImage {
  /// Resizes image and converts to CVPixelBuffer
  func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
           kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                     kCVPixelFormatType_32BGRA, attrs, &pixelBuffer)
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
    CVPixelBufferLockBaseAddress(buffer, [])
    let context = CIContext()
    // Use CIImage for easy resizing handling
    if let ciImage = CIImage(image: self) {
      let scaleX = CGFloat(width) / size.width
      let scaleY = CGFloat(height) / size.height
      let resized = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
      // Render into the pixel buffer
      context.render(resized, to: buffer)
    }
    CVPixelBufferUnlockBaseAddress(buffer, [])
    return buffer
  }
}