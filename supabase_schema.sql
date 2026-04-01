-- ============================================================
-- OLEA Insurance AI — Complete Supabase Schema
-- Run this ENTIRE script in your Supabase SQL Editor
-- ============================================================

-- ============================================================
-- 1. PROFILES TABLE (linked to Supabase Auth users)
-- ============================================================
CREATE TABLE IF NOT EXISTS profiles (
  id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  first_name TEXT NOT NULL,
  last_name  TEXT NOT NULL,
  profession TEXT NOT NULL,
  income     NUMERIC NOT NULL CHECK (income >= 0),
  social_status TEXT NOT NULL CHECK (social_status IN ('single','married','divorced','widowed')),
  kids       INTEGER NOT NULL DEFAULT 0 CHECK (kids >= 0),
  cars       INTEGER NOT NULL DEFAULT 1 CHECK (cars >= 0),
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Users can read own profile" ON profiles;
CREATE POLICY "Users can read own profile" ON profiles FOR SELECT USING (auth.uid() = id);
DROP POLICY IF EXISTS "Users can insert own profile" ON profiles;
CREATE POLICY "Users can insert own profile" ON profiles FOR INSERT WITH CHECK (auth.uid() = id);
DROP POLICY IF EXISTS "Users can update own profile" ON profiles;
CREATE POLICY "Users can update own profile" ON profiles FOR UPDATE USING (auth.uid() = id);


-- ============================================================
-- 2. HISTORY TABLE (chat message storage per session)
-- Required by main.py InsuranceChatbot._load_memory()
-- ============================================================
DROP TABLE IF EXISTS history CASCADE;
CREATE TABLE history (
  id           BIGSERIAL PRIMARY KEY,
  session_id   TEXT NOT NULL,
  user_input   TEXT NOT NULL,
  response     TEXT NOT NULL,
  mode         TEXT DEFAULT 'text',
  sources      JSONB DEFAULT '[]',
  timestamp    TEXT DEFAULT to_char(now(), 'YYYY-MM-DD HH24:MI:SS')
);

-- Index for fast session lookups
CREATE INDEX IF NOT EXISTS idx_history_session ON history(session_id);

-- Allow the service role full access (used by backend)
ALTER TABLE history ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Service role full access to history" ON history;
CREATE POLICY "Service role full access to history"
  ON history FOR ALL
  USING (true)
  WITH CHECK (true);


-- ============================================================
-- 3. CONVERSATIONS TABLE (sidebar conversation list)
-- Required by InsuranceChatbot.get_all_conversations()
-- ============================================================
DROP TABLE IF EXISTS conversations CASCADE;
CREATE TABLE conversations (
  id         TEXT PRIMARY KEY,
  title      TEXT NOT NULL DEFAULT 'New Conversation',
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Service role full access to conversations" ON conversations;
CREATE POLICY "Service role full access to conversations"
  ON conversations FOR ALL
  USING (true)
  WITH CHECK (true);


-- ============================================================
-- 4. REMINDER: Enable Email Confirmation
-- Authentication > Providers > Email > "Confirm email" = ON
-- ============================================================
