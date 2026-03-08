//
//  Constants.template.swift
//  InfineonProject
//
//  Created by Aaron Ma on 1/13/26.
//
//  Legacy fallback only:
//  Runtime config now prefers process env / Info.plist keys injected from build settings.
//  Keep this file for backward compatibility when build-time injection is not configured.
//
//  SUPABASE (for Authentication):
//  ==============================
//  1. Go to https://supabase.com and create a project (or use existing)
//  2. Navigate to Project Settings > API
//  3. Copy the "Project URL" and paste it as supabaseURL below
//  4. Copy the "anon public" key and paste it as supabaseAnonKey below

//import SwiftUI

//enum Constants {
//    enum Supabase {
//        /// Your Supabase project URL
//        /// Found at: Project Settings > API > Project URL
//        static let supabaseURL = "YOUR_SUPABASE_URL_HERE"
//
//        /// Your Supabase publishable (public) key
//        /// Found at: Project Settings > API > anon public
//        static let supabasePublishableKey = "YOUR_SUPABASE_ANON_KEY_HERE"
//    }
//
//    enum HomeRouteAnnouncer: String {
//        case trips = "_trips"
//        case _tripDetail = "_tripDetail"
//    }
//}
