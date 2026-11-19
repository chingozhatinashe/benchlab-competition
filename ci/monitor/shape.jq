# The shape of a JSON document, never its values: one line per field path (array indices
# collapsed to []), with its type(s), "link" for strings that look like URLs, and the
# lengths of arrays. Key names that look like links or query strings are masked.
# Used by .github/workflows/monitor.yml on the Benchlab portal's response, whose values
# include presigned links that must not reach a public log.
def safe_key: if test("://|[?&=]|^.{48,}") then "<masked key>" else . end;
def path_name($p): $p | map(if type == "number" then "[]" else (tostring | safe_key) end)
                  | join(".") | if . == "" then "." else . end;
def kind: if type == "string" and test("^https?://") then "link"
          elif type == "array" then "array(\(length))"
          else type end;
[paths as $p | {path: path_name($p), kind: (getpath($p) | kind)}]
| group_by(.path)
| .[]
| "\(.[0].path): \([.[].kind] | unique
    | (map(select(startswith("array(") | not))
       + (map(select(startswith("array(")) | ltrimstr("array(") | rtrimstr(")") | tonumber)
          | if length > 0 then ["array(len \(min)..\(max))"] else [] end))
    | join(" | "))"
