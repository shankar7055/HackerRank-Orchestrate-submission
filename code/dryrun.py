#!/usr/bin/env python3
"""Dry-run: exercises all pre-API stages of main.py without needing GROQ_API_KEY."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import main as m

data = m.load_all()
idx  = m.build_indexes(data)
msgs = data["messages"]

print("Data loaded OK")
print(f"  messages            : {len(msgs)}")
print(f"  business idx        : {len(idx['business'])}")
print(f"  group_member idx    : {len(idx['group_member'])} groups")
print(f"  events_by_user_msg  : {len(idx['events_by_user_msg'])} (user,msg) pairs")
print(f"  images idx          : {len(idx['images'])}")
print(f"  voice_notes idx     : {len(idx['voice_notes'])}")
print(f"  valid_history_ids   : {len(idx['valid_history_ids'])}")

# Voice note index + path resolution
voice_msgs = [msg for msg in msgs if msg.get("media_type") == "voice"]
print(f"\nVoice messages ({len(voice_msgs)}):")
for msg in voice_msgs:
    mid = msg.get("media_id", "")
    hit = idx["voice_notes"].get(mid)
    if hit:
        path = m.DATASET_DIR / hit["file_path"]
        print(f"  {msg['message_id']:12s}  {mid:8s}  {hit['file_path']}  exists={path.exists()}")
    else:
        print(f"  {msg['message_id']:12s}  {mid:8s}  MISSING from voice_notes index")

# Image index + path resolution
image_msgs = [msg for msg in msgs if msg.get("media_type") == "image"]
print(f"\nImage messages ({len(image_msgs)}) — first 5:")
for msg in image_msgs[:5]:
    mid = msg.get("media_id", "")
    hit = idx["images"].get(mid)
    if hit:
        path = m.DATASET_DIR / hit["file_path"]
        print(f"  {msg['message_id']:12s}  {mid:8s}  {hit['file_path']}  exists={path.exists()}")
    else:
        print(f"  {msg['message_id']:12s}  {mid:8s}  MISSING from images index")

# Pre-gate on all 110
gate_hits, gate_pass = [], []
for msg in msgs:
    r = m.pre_gate(msg)
    if r:
        gate_hits.append((msg["message_id"], r["_gate"]))
    else:
        gate_pass.append(msg["message_id"])

print(f"\nPre-gate: {len(gate_hits)} blocked, {len(gate_pass)} pass to LLM")
for mid, tag in gate_hits:
    print(f"  {mid:12s}  [{tag}]")

# Context assembly on first 5 gate-pass messages
print(f"\nContext assembly (first 5 gate-pass):")
for mid in gate_pass[:5]:
    msg = next(m2 for m2 in msgs if m2["message_id"] == mid)
    ctx = m.assemble_context(msg, idx, data)
    ev  = m.retrieve_evidence(msg, idx, top_k=3)
    print(f"  {mid:12s}  muted={ctx['group_muted']}  mentioned={ctx['mentioned']}  "
          f"sender_trust={ctx['sender_trust']}  biz_verified={ctx['business_verified']}  "
          f"opted_out={ctx['opted_out']}  active_order={ctx['active_order']}  "
          f"evidence={len(ev)}")

print("\nDry-run complete — all pre-API stages OK")
