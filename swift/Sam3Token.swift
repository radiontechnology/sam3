//
// Sam3Token.swift
// meesma_cam
//
// Created by GBORRAS on 13/1/26.
//
import Foundation
public class SimpleTokenizer {
  // MARK: - Properties
  private let byteEncoder: [UInt8: String]
  private let byteDecoder: [String: UInt8]
  private var encoder: [String: Int] = [:]
  private var decoder: [Int: String] = [:]
  private var bpeRanks: [[String]: Int] = [:]
  private var cache: [String: String] = [:]
  private let regexPattern: NSRegularExpression
  public let contextLength: Int
  public let sotTokenId: Int
  public let eotTokenId: Int
  // MARK: - Initialization
  /// Initializes the tokenizer.
  /// - Parameters:
  ///  - bpeContent: The raw string content of the BPE merge file (e.g. `bpe_simple_vocab_16e6.txt`).
  ///  - contextLength: The context length (default is 77 for CLIP).
  public init(bpeContent: String, contextLength: Int = 77) {
    self.contextLength = contextLength
    // 1. Setup Byte Encoder/Decoder
    self.byteEncoder = SimpleTokenizer.bytesToUnicode()
    self.byteDecoder = Dictionary(uniqueKeysWithValues: byteEncoder.map { ($1, $0) })
    // 2. Parse BPE Merges
    var merges = bpeContent.components(separatedBy: "\n")
    // Python slice: merges = merges[1 : 49152 - 256 - 2 + 1]
    let start = 1
    let end = 49152 - 256 - 2 + 1
    if merges.count > end {
      merges = Array(merges[start..<end])
    } else {
      // Fallback for smaller/test files
      merges = Array(merges.dropFirst())
    }
    let mergeTuples: [[String]] = merges.compactMap { line in
      let parts = line.split(separator: " ")
      if parts.count == 2 {
        return parts.map { String($0) }
      }
      return nil
    }
    // 3. Build Vocabulary
    var vocab: [String] = Array(byteEncoder.values)
    vocab = vocab + vocab.map { $0 + "</w>" }
    for merge in mergeTuples {
      vocab.append(merge.joined())
    }
    let specialTokens = ["<start_of_text>", "<end_of_text>"]
    vocab.append(contentsOf: specialTokens)
    self.encoder = Dictionary(uniqueKeysWithValues: zip(vocab, 0..<vocab.count))
    self.decoder = Dictionary(uniqueKeysWithValues: zip(0..<vocab.count, vocab))
    // Rank dictionary for BPE
    for (index, pair) in mergeTuples.enumerated() {
      self.bpeRanks[pair] = index
    }
    // Cache special tokens
    for token in specialTokens {
      self.cache[token] = token
    }
    self.sotTokenId = encoder["<start_of_text>"]!
    self.eotTokenId = encoder["<end_of_text>"]!
    // 4. Compile Regex
    // Pattern logic: <start>|<end>|'s|'t... etc
    let specialPattern = specialTokens.map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "|")
    let patternStr = specialPattern + #"|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#
    self.regexPattern = try! NSRegularExpression(pattern: patternStr, options: [.caseInsensitive])
  }
  // MARK: - Public API
  /// Returns the tokenized representation of given input string(s)
  /// Returns a two-dimensional array [BatchSize, ContextLength]
  public func tokenize(_ texts: [String]) -> [[Int]] {
    var result: [[Int]] = []
    for text in texts {
      let encoded = encode(text)
      var tokens = [sotTokenId] + encoded + [eotTokenId]
      // Truncate
      if tokens.count > contextLength {
        tokens = Array(tokens.prefix(contextLength))
        tokens[contextLength - 1] = eotTokenId
      }
      // Pad (using 0 as padding, though CLIP 0 is technically '!', it is standard for tensor padding)
      while tokens.count < contextLength {
        tokens.append(0)
      }
      result.append(tokens)
    }
    return result
  }
  public func tokenize(_ text: String) -> [[Int]] {
    return tokenize([text])
  }
  public func decode(_ tokens: [Int]) -> String {
    let textParts = tokens.compactMap { decoder[$0] }
    let joined = textParts.joined()
    // Map back from unicode safe chars to bytes
    var byteArray: [UInt8] = []
    for char in joined {
      let strChar = String(char)
      if let byte = byteDecoder[strChar] {
        byteArray.append(byte)
      }
    }
    if let decodedString = String(bytes: byteArray, encoding: .utf8) {
       return decodedString.replacingOccurrences(of: "</w>", with: " ")
    }
    return ""
  }
  // MARK: - Internal Logic
  private func encode(_ text: String) -> [Int] {
    var bpeTokens: [Int] = []
    // Clean text
    let cleanedText = whitespaceClean(basicClean(text)).lowercased()
    // Find all regex matches
    let range = NSRange(cleanedText.startIndex..<cleanedText.endIndex, in: cleanedText)
    let matches = regexPattern.matches(in: cleanedText, options: [], range: range)
    for match in matches {
      if let range = Range(match.range, in: cleanedText) {
        let tokenStr = String(cleanedText[range])
        // Map utf8 bytes to specific unicode chars used in vocab
        let utf8Bytes = Array(tokenStr.utf8)
        let mappedToken = utf8Bytes.map { byteEncoder[$0]! }.joined()
        // Apply BPE
        let bpeResult = bpe(mappedToken)
        // Map parts to IDs
        for part in bpeResult.split(separator: " ") {
          if let id = encoder[String(part)] {
            bpeTokens.append(id)
          }
        }
      }
    }
    return bpeTokens
  }
  private func bpe(_ token: String) -> String {
    if let cached = cache[token] {
      return cached
    }
    var word: [String] = token.dropLast().map { String($0) }
    word.append(String(token.last!) + "</w>")
    var pairs = getPairs(word)
    if pairs.isEmpty {
      return token + "</w>"
    }
    while true {
      // Find the pair with the lowest rank (lowest index in merges)
      var minPair: [String]? = nil
      var minRank = Int.max
      for pair in pairs {
        if let rank = bpeRanks[pair] {
          if rank < minRank {
            minRank = rank
            minPair = pair
          }
        }
      }
      guard let bigram = minPair else { break }
      let first = bigram[0]
      let second = bigram[1]
      var newWord: [String] = []
      var i = 0
      while i < word.count {
        // Find occurrence of `first`
        if let j = word[i...].firstIndex(of: first) {
          newWord.append(contentsOf: word[i..<j])
          i = j
        } else {
          newWord.append(contentsOf: word[i...])
          break
        }
        if word[i] == first && i < word.count - 1 && word[i + 1] == second {
          newWord.append(first + second)
          i += 2
        } else {
          newWord.append(word[i])
          i += 1
        }
      }
      word = newWord
      if word.count == 1 { break }
      pairs = getPairs(word)
    }
    let result = word.joined(separator: " ")
    cache[token] = result
    return result
  }
  // MARK: - Private Helpers
  private func getPairs(_ word: [String]) -> Set<[String]> {
    var pairs = Set<[String]>()
    if word.isEmpty { return pairs }
    var prevChar = word[0]
    for i in 1..<word.count {
      let char = word[i]
      pairs.insert([prevChar, char])
      prevChar = char
    }
    return pairs
  }
  private func basicClean(_ text: String) -> String {
    // 1. Fix text (Normalization)
    var cleaned = text.precomposedStringWithCanonicalMapping
    // 2. HTML Unescape (Using NSAttributedString)
    if let data = cleaned.data(using: .utf8) {
      let options: [NSAttributedString.DocumentReadingOptionKey: Any] = [
        .documentType: NSAttributedString.DocumentType.html,
        .characterEncoding: String.Encoding.utf8.rawValue
      ]
      if let attributedString = try? NSAttributedString(data: data, options: options, documentAttributes: nil) {
        cleaned = attributedString.string
      }
    }
    return cleaned.trimmingCharacters(in: .whitespaces)
  }
  private func whitespaceClean(_ text: String) -> String {
    let regex = try! NSRegularExpression(pattern: "\\s+")
    let range = NSRange(text.startIndex..<text.endIndex, in: text)
    let cleaned = regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: " ")
    return cleaned.trimmingCharacters(in: .whitespaces)
  }
  private static func bytesToUnicode() -> [UInt8: String] {
    var bs: [Int] = []
    bs.append(contentsOf: Array(ord("!")...ord("~")))
    bs.append(contentsOf: Array(ord("¡")...ord("¬")))
    bs.append(contentsOf: Array(ord("®")...ord("ÿ")))
    var cs = bs
    var n = 0
    for b in 0..<256 {
      if !bs.contains(b) {
        bs.append(b)
        cs.append(256 + n)
        n += 1
      }
    }
    var mapping: [UInt8: String] = [:]
    for (index, b) in bs.enumerated() {
      if let scalar = UnicodeScalar(cs[index]) {
        mapping[UInt8(b)] = String(scalar)
      }
    }
    return mapping
  }
  private static func ord(_ s: String) -> Int {
    return Int(s.unicodeScalars.first!.value)
  }
}