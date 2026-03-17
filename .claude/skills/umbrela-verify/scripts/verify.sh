#!/usr/bin/env bash
# umbrela-verify: Validate umbrela judge or qrel output artifacts.
#
# Usage:
#   bash verify.sh <artifact-path> [judge|qrel]
#
# If type is omitted, auto-detects based on file content.

set -euo pipefail

ARTIFACT_PATH="${1:?Usage: verify.sh <artifact-path> [judge|qrel]}"
ARTIFACT_TYPE="${2:-auto}"

# Colors (respect NO_COLOR)
if [[ -z "${NO_COLOR:-}" ]]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; NC=''
fi

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; FAILURES=$((FAILURES + 1)); }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

FAILURES=0

# --- Basic file checks ---
echo "=== File Integrity ==="

if [[ ! -f "$ARTIFACT_PATH" ]]; then
  fail "File not found: $ARTIFACT_PATH"
  exit 1
fi
pass "File exists: $ARTIFACT_PATH"

LINE_COUNT=$(wc -l < "$ARTIFACT_PATH" | tr -d ' ')
if [[ "$LINE_COUNT" -eq 0 ]]; then
  fail "File is empty"
  exit 1
fi
pass "File has $LINE_COUNT lines"

# --- Auto-detect type ---
if [[ "$ARTIFACT_TYPE" == "auto" ]]; then
  FIRST_LINE=$(head -1 "$ARTIFACT_PATH")
  if echo "$FIRST_LINE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'judgment' in d" 2>/dev/null; then
    ARTIFACT_TYPE="judge"
  else
    ARTIFACT_TYPE="qrel"
  fi
  pass "Auto-detected type: $ARTIFACT_TYPE"
fi

echo ""
echo "=== Content Validation ($ARTIFACT_TYPE) ==="

if [[ "$ARTIFACT_TYPE" == "judge" ]]; then
  # Validate JSONL judge output
  BAD_LINES=$(python3 -c "
import json, sys
bad = 0
for i, line in enumerate(open('$ARTIFACT_PATH'), 1):
    line = line.strip()
    if not line: continue
    try:
        json.loads(line)
    except json.JSONDecodeError:
        print(f'  Line {i}: invalid JSON', file=sys.stderr)
        bad += 1
print(bad)
")
  if [[ "$BAD_LINES" -eq 0 ]]; then
    pass "All lines are valid JSON"
  else
    fail "$BAD_LINES lines have invalid JSON"
  fi

  python3 -c "
import json, sys

path = '$ARTIFACT_PATH'
failures = 0

records = []
for line in open(path):
    line = line.strip()
    if line:
        records.append(json.loads(line))

total = len(records)
parse_ok = sum(1 for r in records if r.get('result_status') == 1)
parse_fail = total - parse_ok

# Label range check
bad_labels = 0
for i, r in enumerate(records):
    j = r.get('judgment')
    if not isinstance(j, int) or j < 0 or j > 3:
        print(f'✗ Record {i+1}: judgment={j} out of range 0-3')
        bad_labels += 1
        failures += 1
if bad_labels == 0:
    print(f'✓ All {total} records have valid judgment labels (0-3)')

# Required fields
required = ['model', 'query', 'passage', 'judgment', 'result_status']
for i, r in enumerate(records):
    for field in required:
        if field not in r:
            print(f'✗ Record {i+1}: missing field \"{field}\"')
            failures += 1

# Model consistency
models = set(r.get('model', '') for r in records)
if len(models) == 1:
    print(f'✓ Consistent model: {models.pop()}')
elif len(models) > 1:
    print(f'⚠ Multiple models found: {models}')

# Parse success rate
rate = parse_ok / total * 100 if total > 0 else 0
print(f'✓ Parse success rate: {parse_ok}/{total} ({rate:.1f}%)')
if parse_fail / total > 0.10 if total > 0 else False:
    print(f'⚠ High parse failure rate: {parse_fail}/{total} records fell back to label 0')

# Duplicate check
pairs = [(r.get('query',''), r.get('passage','')) for r in records]
dupes = len(pairs) - len(set(pairs))
if dupes > 0:
    print(f'⚠ {dupes} duplicate query-passage pair(s)')
else:
    print(f'✓ No duplicate query-passage pairs')

# Label distribution
from collections import Counter
dist = Counter(r.get('judgment') for r in records)
print(f'✓ Label distribution: ' + ', '.join(f'{k}={v}' for k, v in sorted(dist.items())))

if failures == 0:
    print('✓ All records are well-formed')
sys.exit(1 if failures > 0 else 0)
" 2>&1 || FAILURES=$((FAILURES + 1))

elif [[ "$ARTIFACT_TYPE" == "qrel" ]]; then
  # Validate TREC qrel format
  python3 -c "
import sys

path = '$ARTIFACT_PATH'
failures = 0
line_count = 0

for i, line in enumerate(open(path), 1):
    line = line.strip()
    if not line:
        continue
    line_count += 1
    parts = line.split()
    if len(parts) < 4:
        print(f'✗ Line {i}: expected >= 4 fields, got {len(parts)}')
        failures += 1
        continue
    qid, q0, docid, label = parts[0], parts[1], parts[2], parts[3]
    try:
        label_int = int(label)
        if label_int < 0 or label_int > 3:
            print(f'✗ Line {i}: label {label} out of range 0-3')
            failures += 1
    except ValueError:
        print(f'✗ Line {i}: non-integer label \"{label}\"')
        failures += 1

if failures == 0:
    print(f'✓ All {line_count} qrel entries are well-formed')
sys.exit(1 if failures > 0 else 0)
" 2>&1 || FAILURES=$((FAILURES + 1))
fi

# --- Summary ---
echo ""
echo "=== Summary ==="
if [[ "$FAILURES" -eq 0 ]]; then
  pass "All checks passed"
  exit 0
else
  fail "$FAILURES check(s) failed"
  exit 1
fi
