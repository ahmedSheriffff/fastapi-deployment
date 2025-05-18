-- Create a table to store API keys
CREATE TABLE api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    active BOOLEAN DEFAULT TRUE
);
-- Insert initial API keys
INSERT INTO api_keys (key, active) VALUES ('770fc0b98ee9ce629fb913c76d01ec6ebe6598f87b438b9128e8cb58e92fc8f5', TRUE);
-- Verify stored API keys
SELECT * FROM api_keys;



