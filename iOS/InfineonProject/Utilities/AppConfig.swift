import Foundation

struct AppConfig {
  struct SupabaseRuntimeConfig {
    let urlString: String?
    let publishableKey: String?
    let sourceDescription: String

    var validationError: String? {
      guard let urlString, !urlString.isEmpty else {
        return "Missing SUPABASE_URL"
      }
      guard let publishableKey, !publishableKey.isEmpty else {
        return "Missing SUPABASE_PUBLISHABLE_KEY"
      }

      let lowerURL = urlString.lowercased()
      let lowerKey = publishableKey.lowercased()
      if lowerURL.contains("placeholder")
        || lowerURL.contains("your-project-id")
        || lowerURL.contains("your_supabase")
      {
        return "SUPABASE_URL is a placeholder value"
      }
      if lowerKey.contains("placeholder")
        || lowerKey.contains("your-supabase")
        || lowerKey.contains("your_supabase")
      {
        return "SUPABASE_PUBLISHABLE_KEY is a placeholder value"
      }

      guard let parsedURL = URL(string: urlString),
        let scheme = parsedURL.scheme?.lowercased(),
        ["https", "http"].contains(scheme),
        parsedURL.host != nil
      else {
        return "SUPABASE_URL is not a valid URL"
      }

      return nil
    }

    var isValid: Bool { validationError == nil }

    var resolvedURL: URL {
      URL(string: urlString ?? "") ?? URL(string: "https://placeholder.supabase.co")!
    }
  }

  static let shared = AppConfig.load()

  let supabase: SupabaseRuntimeConfig

  static func load(
    bundle: Bundle = .main,
    processInfo: ProcessInfo = .processInfo
  ) -> AppConfig {
    let env = processInfo.environment

    let envURL = firstNonEmpty([env["SUPABASE_URL"]])
    let envKey = firstNonEmpty([
      env["SUPABASE_PUBLISHABLE_KEY"],
      env["SUPABASE_KEY"],
      env["SUPABASE_ANON_KEY"],
    ])

    let infoURL = firstNonEmpty([
      bundle.object(forInfoDictionaryKey: "SUPABASE_URL") as? String
    ])
    let infoKey = firstNonEmpty([
      bundle.object(forInfoDictionaryKey: "SUPABASE_PUBLISHABLE_KEY") as? String,
      bundle.object(forInfoDictionaryKey: "SUPABASE_ANON_KEY") as? String,
    ])

    let legacyURL = firstNonEmpty([Constants.Supabase.supabaseURL])
    let legacyKey = firstNonEmpty([Constants.Supabase.supabasePublishableKey])

    let resolvedURL = firstNonEmpty([envURL, infoURL, legacyURL])
    let resolvedKey = firstNonEmpty([envKey, infoKey, legacyKey])

    let sourceDescription: String
    if envURL != nil || envKey != nil {
      sourceDescription = "process-environment"
    } else if infoURL != nil || infoKey != nil {
      sourceDescription = "info-plist"
    } else {
      sourceDescription = "legacy-constants"
    }

    return AppConfig(
      supabase: SupabaseRuntimeConfig(
        urlString: resolvedURL,
        publishableKey: resolvedKey,
        sourceDescription: sourceDescription
      ))
  }

  private static func firstNonEmpty(_ values: [String?]) -> String? {
    for value in values {
      if let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty {
        return trimmed
      }
    }
    return nil
  }
}
