#!/usr/bin/env python3
"""Gap analysis: confidence calibration, cost/runtime, @mention-in-muted-group."""
import csv, re
from collections import defaultdict

def load(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))

msgs   = load('dataset/messages.csv')
sample = load('dataset/sample_messages.csv')
gm     = load('dataset/group_members.csv')
users  = load('dataset/users.csv')
mh     = load('dataset/message_history.csv')
me     = load('dataset/message_events.csv')
biz    = load('dataset/business_accounts.csv')
ubh    = load('dataset/user_business_history.csv')

mute_map = {(r['group_id'], r['user_id']): int(r['group_muted_by_user']) for r in gm}
role_map = {(r['group_id'], r['user_id']): r['role'] for r in gm}
ev_map   = {(r['user_id'], r['message_id']): r for r in me}
biz_map  = {b['business_id']: b for b in biz}
ubh_map  = {(r['user_id'], r['business_id']): r for r in ubh}
mh_by_user = defaultdict(list)
for r in mh:
    mh_by_user[r['user_id']].append(r)
s_map = {s['message_id']: s for s in sample}

# ═══════════════════════════════════════════════════════════════════════════════
# GAP 1 — Confidence calibration
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("GAP 1 — CONFIDENCE CALIBRATION")
print("=" * 72)
print()
print("Proposed Regime B: base=0.50 + 0.15×sender_trust + 0.15×history_match")
print("                   + 0.10×evidence≥1 + 0.05×mention_or_direct")
print("                   - 0.20×poor_media   capped [0.50, 0.90]")
print("Regime A (deterministic gate):  fixed 0.95")
print()

INJECT_RE = re.compile(
    r'routing override|assistant instruction|system note.*router'
    r'|ignore.*routing|action=notify|internal router|mark.*notify', re.I)
OTP_RE = re.compile(
    r'\botp\b|share.*otp|send.*otp|verify.*now.*or.*block'
    r'|account.*block.*otp|profile.*restrict', re.I)

def regime_a(s):
    text = s.get('message_text') or ''
    return s['message_type'] in ('scam','spam') and (
        bool(INJECT_RE.search(text)) or bool(OTP_RE.search(text)))

def calc_b(s):
    uid   = s['user_id']
    ctype = s['conversation_type']
    bid   = s.get('business_id') or ''
    gid   = s.get('group_id') or ''
    media = s.get('media_type') or ''
    fwd   = int(s.get('forwarded_count') or 0)
    ev    = s.get('evidence_message_ids') or 'none'

    # sender_trust
    if ctype == 'group' and gid:
        sid_u = s.get('sender_user_id','')
        trust = 1 if role_map.get((gid, sid_u), 'member') == 'admin' else 0
    elif ctype == 'business' and bid:
        b = biz_map.get(bid, {})
        trust = 1 if (b.get('verified','0')=='1'
                      and b.get('official_domain','') == b.get('domain_used_by_sender','')) else 0
    else:  # personal — always known sender
        trust = 1

    # history_match: past message from same sender/business exists
    past = mh_by_user.get(uid, [])
    hist = any(r.get('business_id')==bid or r.get('sender_user_id')==s.get('sender_user_id','__')
               for r in past)

    evidence   = 0 if ev in ('none','',None) else 1
    mention    = 1 if (f'@{uid}' in (s.get('message_text') or '').lower()
                       or ctype == 'personal') else 0
    poor_media = 1 if (media in ('voice','image') and fwd > 3) else 0

    sigs = dict(trust=trust, hist=int(hist), ev=evidence, mention=mention, poor=poor_media)
    val  = 0.50 + 0.15*trust + 0.15*hist + 0.10*evidence + 0.05*mention - 0.20*poor_media
    return sigs, round(min(0.90, max(0.50, val)), 2)

target_ids = [
    'sample_msg_001',  # notify  urgent         admin sender    has evidence
    'sample_msg_004',  # notify  business_update verified biz   has evidence
    'sample_msg_005',  # notify  event           verified+hist  has evidence
    'sample_msg_007',  # digest  promotion       opted-in       has evidence
    'sample_msg_009',  # digest  greeting        harmless       has evidence
    'sample_msg_013',  # mute    greeting        repeat fwder   has evidence
    'sample_msg_050',  # digest  personal        known contact  has evidence
    'sample_msg_049',  # digest  unknown         unfamiliar     evidence=none
    'sample_msg_019',  # mute    scam            OTP phish      has evidence
    'sample_msg_053',  # mute    scam            inject         has evidence
]

print(f"{'sample_id':22s} {'act':7s} {'type':16s} {'GT':5s} {'CALC':5s} {'delta':6s}  signals")
print("-" * 105)
deltas = []
for sid in target_ids:
    s = s_map.get(sid)
    if not s:
        continue
    gt = float(s['confidence'])
    if regime_a(s):
        calc = 0.95
        sigs_str = "REGIME_A"
    else:
        sigs, calc = calc_b(s)
        sigs_str = ' '.join(f"{k}={v}" for k,v in sigs.items())
    delta = calc - gt
    deltas.append(delta)
    print(f"  {sid:22s} {s['action']:7s} {s['message_type']:16s} {gt:5.2f} {calc:5.2f} {delta:+6.2f}  {sigs_str}")

non_a = [(s['message_id'], *calc_b(s)[-1:], float(s['confidence']))
         for s in sample if not regime_a(s)]
# rebuild with actual values
non_a_full = []
for s in sample:
    if not regime_a(s):
        sigs, calc = calc_b(s)
        gt = float(s['confidence'])
        non_a_full.append((s['message_id'], calc, gt, calc-gt))
non_a_full.sort(key=lambda x: x[1])

print()
print(f"Mean Δ across 8 hand-checked rows: {sum(deltas)/len(deltas):+.3f}")
print(f"Mean |Δ|: {sum(abs(d) for d in deltas)/len(deltas):.3f}")
print()
print("All non-Regime-A sample rows, sorted by calc (lowest first):")
print(f"  {'msg_id':22s} {'calc':5s} {'gt':5s} {'delta':6s}")
for mid, calc, gt, d in non_a_full:
    sign = '+' if d >= 0 else ''
    print(f"  {mid:22s} {calc:.2f}  {gt:.2f}  {sign}{d:.2f}")

gt_vals = [float(s['confidence']) for s in sample]
print(f"\nObserved GT range: {min(gt_vals):.2f} – {max(gt_vals):.2f}  avg={sum(gt_vals)/len(gt_vals):.2f}")
calcs = [x[1] for x in non_a_full]
print(f"Formula output range (non-A): {min(calcs):.2f} – {max(calcs):.2f}  avg={sum(calcs)/len(calcs):.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# GAP 2 — Cost / runtime
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("GAP 2 — COST / RUNTIME ESTIMATE")
print("=" * 72)
print()

GATE_OTP  = re.compile(
    r'\botp\b|share.*otp|send.*otp|account.*block|profile.*restrict'
    r'|verify.*account.*now|verify.*now.*or', re.I)
GATE_INJ  = INJECT_RE
GATE_CHAIN = re.compile(
    r'forward.*bless|share.*10 people|send.*family group|chain'
    r'|luck.*when you share|good luck|share.*group', re.I)

blocked, passed = [], []
for m in msgs:
    text = m.get('message_text') or ''
    bid  = m.get('business_id') or ''
    fwd  = int(m.get('forwarded_count') or 0)
    tags = []
    if GATE_INJ.search(text):  tags.append('inject')
    if GATE_OTP.search(text):  tags.append('otp')
    if fwd >= 7 and GATE_CHAIN.search(text): tags.append('chain')
    if bid:
        b = biz_map.get(bid, {})
        if (b.get('verified','1')=='0'
            and b.get('official_domain','')
            and b.get('official_domain','') != b.get('domain_used_by_sender','')):
            tags.append('domain_phish')
    if tags:
        blocked.append((m['message_id'], tags))
    else:
        passed.append(m)

print(f"Pre-gate:")
print(f"  Hard-blocked : {len(blocked)}")
for mid, tags in blocked:
    print(f"    {mid:12s}  [{', '.join(tags)}]")
print(f"  Pass to LLM  : {len(passed)}")

p_ids = {m['message_id'] for m in passed}
imgs_llm  = [m for m in msgs if m['message_id'] in p_ids and m.get('media_type')=='image']
voice_llm = [m for m in msgs if m['message_id'] in p_ids and m.get('media_type')=='voice']
print(f"\nMedia reaching LLM: {len(imgs_llm)} images, {len(voice_llm)} voice")
print(f"All images need OCR: 15   All voice need ASR: 8 (gate result doesn't skip media)")

print()
print("Runtime (sequential, single-threaded):")
print(f"  regex gate     0.5ms × 110            = {0.5*110/1000:.2f}s")
print(f"  OCR (vision)   1.2s  × 15 images      = {1.2*15:.1f}s  (GPT-4o / Claude vision API)")
print(f"  ASR (whisper)  2.0s  × 8 voice notes  = {2.0*8:.1f}s  (OpenAI whisper-1 API)")
print(f"  LLM classify   1.5s  × {len(passed)} msgs       = {1.5*len(passed):.1f}s  (GPT-4o-mini or Haiku)")
seq = 0.055 + 1.2*15 + 2.0*8 + 1.5*len(passed)
print(f"  ─────────────────────────────────────────────")
print(f"  Total sequential                       = {seq:.1f}s  ({seq/60:.1f} min)")

par = 0.055 + max(1.2*15, 2.0*8) + 1.5*len(passed)
print(f"\nRuntime (OCR ‖ ASR concurrent):")
print(f"  = {par:.1f}s  ({par/60:.1f} min)")
print(f"\nHackathon budget: 24h.  Estimated run: < 5 min.  No constraint.")


# ═══════════════════════════════════════════════════════════════════════════════
# GAP 3 — @mention in muted group
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("GAP 3 — @MENTION IN MUTED GROUP")
print("=" * 72)
print()

def find_muted_mention(rows, label):
    found = []
    for m in rows:
        if m.get('conversation_type') != 'group':
            continue
        uid, gid = m['user_id'], m.get('group_id','')
        if not mute_map.get((gid, uid), 0):
            continue
        text = m.get('message_text') or ''
        if f'@{uid}' in text:
            found.append(m)
    print(f"{label} ({len(rows)} rows):")
    if found:
        for m in found:
            print(f"  ✓ FOUND: {m['message_id']}  user={m['user_id']}  group={m['group_id']}")
            print(f"    muted={mute_map.get((m['group_id'],m['user_id']),0)}")
            print(f"    text: {(m.get('message_text') or '')[:120]}")
    else:
        print("  (none)")
    return found

f1 = find_muted_mention(msgs,   "messages.csv")
f2 = find_muted_mention(sample, "sample_messages.csv")
f3 = find_muted_mention(mh,     "message_history.csv")

print()
print("All muted-group messages in messages.csv (regardless of @mention):")
print(f"  {'msg_id':12s} {'user':7s} {'group':12s} @mention  fwd  text[:70]")
print("  " + "-"*90)
for m in msgs:
    if m.get('conversation_type') != 'group':
        continue
    uid, gid = m['user_id'], m.get('group_id','')
    if not mute_map.get((gid, uid), 0):
        continue
    text = m.get('message_text') or ''
    mention = f'@{uid}' in text
    print(f"  {m['message_id']:12s} {uid:7s} {gid:12s} {str(mention):9s} "
          f"{m['forwarded_count']:4s}  {text[:70]}")

print()
if not (f1 or f2 or f3):
    print("CONCLUSION: No @mention-in-muted-group row exists anywhere in the dataset.")
    print("  This case must be handled by stated rule, not observed example.")
    print("  Stated rule: @mention of receiving user in a muted group should")
    print("  override the mute signal — route to notify or digest depending on")
    print("  urgency/sender trust, never force-mute solely because group is muted.")
else:
    print("CONCLUSION: Live example found — see above for ground truth.")

if __name__ == '__main__':
    pass
