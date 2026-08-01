#!/usr/bin/env python3
"""
Quick dataset profiling script using only standard library (no pandas dependency)
"""
import csv
import json
from collections import Counter, defaultdict

def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def analyze():
    print("=== LOADING DATA ===\n")
    messages = load_csv('dataset/messages.csv')
    sample = load_csv('dataset/sample_messages.csv')
    group_members = load_csv('dataset/group_members.csv')
    business = load_csv('dataset/business_accounts.csv')
    
    print(f"Loaded {len(messages)} messages to classify")
    print(f"Loaded {len(sample)} sample messages with solutions")
    print(f"Loaded {len(group_members)} group memberships")
    print(f"Loaded {len(business)} business accounts\n")
    
    # Check schema matches problem statement
    print("=== SCHEMA VERIFICATION ===\n")
    msg_cols = list(messages[0].keys()) if messages else []
    sample_cols = list(sample[0].keys()) if sample else []
    
    print("messages.csv columns:", msg_cols)
    print("\nsample_messages.csv has EXTRA output columns:", 
          [c for c in sample_cols if c not in msg_cols])
    print("(These are what we need to predict)\n")
    
    # Profile messages
    print("=== MESSAGE PROFILE ===\n")
    conv_types = Counter(m['conversation_type'] for m in messages)
    media_types = Counter(m.get('media_type', 'text') or 'text' for m in messages)
    
    # Count forwarded
    forwarded = sum(1 for m in messages if m.get('forwarded_count') and int(m['forwarded_count']) > 0)
    high_forward = [m['message_id'] for m in messages if m.get('forwarded_count') and int(m['forwarded_count']) >= 7]
    
    print(f"Conversation types: {dict(conv_types)}")
    print(f"Media types: {dict(media_types)}")
    print(f"Forwarded messages: {forwarded}/{len(messages)} ({100*forwarded/len(messages):.1f}%)")
    print(f"High forward count (>=7): {len(high_forward)} messages")
    print(f"  Examples: {high_forward[:5]}\n")
    
    # Check domain mismatches in business accounts
    print("=== BUSINESS DOMAIN MISMATCHES (Phishing Signals) ===\n")
    mismatches = []
    for b in business:
        official = b.get('official_domain', '')
        used = b.get('domain_used_by_sender', '')
        if official and used and official != used:
            mismatches.append({
                'business_id': b['business_id'],
                'name': b['display_name'],
                'verified': b['verified'],
                'official': official,
                'used': used,
                'reports': b['user_reports_30d']
            })
    
    print(f"Found {len(mismatches)} businesses with domain mismatch:")
    for m in mismatches[:10]:
        print(f"  {m['business_id']:15} {m['name']:30} verified={m['verified']} reports={m['reports']:3}")
        print(f"    Official: {m['official']:40} Used: {m['used']}")
    
    # Check group mute patterns
    print("\n=== USER MUTE PATTERNS ===\n")
    muted_by_user = defaultdict(int)
    for gm in group_members:
        if gm.get('group_muted_by_user') == '1':
            muted_by_user[gm['user_id']] += 1
    
    top_muters = sorted(muted_by_user.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Users who muted most groups:")
    for user, count in top_muters:
        print(f"  {user}: {count} muted groups")
    
    # Show sample output format
    print("\n=== SAMPLE OUTPUT FORMAT ===\n")
    for i, row in enumerate(sample[:3]):
        print(f"Example {i+1}:")
        print(f"  message_id: {row['message_id']}")
        print(f"  action: {row['action']}")
        print(f"  message_type: {row['message_type']}")
        print(f"  reason: {row['reason'][:80]}...")
        print(f"  confidence: {row['confidence']}")
        print(f"  evidence_message_ids: {row['evidence_message_ids']}")
        print()
    
    # Check scam patterns in messages
    print("=== POTENTIAL SCAM PATTERNS ===\n")
    scam_keywords = ['otp', 'verify', 'urgent', 'blocked', 'expire', 'immediately', 'routing override']
    potential_scams = []
    for m in messages:
        text = (m.get('message_text') or '').lower()
        if any(kw in text for kw in scam_keywords):
            potential_scams.append(m['message_id'])
    
    print(f"Messages with scam keywords: {len(potential_scams)}")
    print(f"  Examples: {potential_scams[:10]}\n")

if __name__ == '__main__':
    analyze()
