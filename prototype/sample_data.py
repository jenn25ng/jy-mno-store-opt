"""
매장수 최적화 — 샘플 데이터 생성 + 폐점 스코어 계산 프로토타입
실제 데이터 연결 전, 현실적인 샘플로 스코어 로직을 검증하는 용도
"""

import json, math, random
random.seed(42)

# ── 실제 행정구역 기반 상권 (시구군 수준) ──────────────────────────────
DISTRICTS = [
    # (구/군명, 위도, 경도, 추정인구, 상권유형)
    ("서울 강남구",    37.5172, 127.0473, 540000, "도심"),
    ("서울 서초구",    37.4837, 127.0324, 430000, "도심"),
    ("서울 마포구",    37.5663, 126.9014, 390000, "도심"),
    ("서울 송파구",    37.5145, 127.1059, 670000, "도심"),
    ("서울 영등포구",  37.5264, 126.8961, 380000, "도심"),
    ("서울 노원구",    37.6542, 127.0568, 520000, "외곽"),
    ("서울 관악구",    37.4784, 126.9516, 490000, "외곽"),
    ("서울 은평구",    37.6026, 126.9291, 480000, "외곽"),
    ("서울 강북구",    37.6397, 127.0257, 320000, "외곽"),
    ("서울 도봉구",    37.6688, 127.0471, 330000, "외곽"),
    ("인천 남동구",    37.4469, 126.7314, 530000, "수도권"),
    ("인천 부평구",    37.5075, 126.7218, 500000, "수도권"),
    ("경기 수원시",    37.2636, 127.0286, 1200000,"수도권"),
    ("경기 성남시",    37.4200, 127.1267, 920000, "수도권"),
    ("경기 안양시",    37.3943, 126.9568, 580000, "수도권"),
    ("경기 용인시",    37.2411, 127.1776, 1070000,"수도권"),
    ("경기 부천시",    37.5034, 126.7660, 830000, "수도권"),
    ("경기 고양시",    37.6584, 126.8320, 1070000,"수도권"),
    ("경기 화성시",    37.1994, 126.8317, 900000, "수도권"),
    ("경기 평택시",    36.9921, 127.1128, 580000, "수도권"),
    ("부산 해운대구",  35.1631, 129.1637, 420000, "지방"),
    ("부산 부산진구",  35.1598, 129.0532, 360000, "지방"),
    ("부산 동래구",    35.2052, 129.0843, 270000, "지방"),
    ("부산 사하구",    35.1040, 128.9745, 320000, "지방"),
    ("대구 수성구",    35.8587, 128.6309, 430000, "지방"),
    ("대구 달서구",    35.8296, 128.5330, 590000, "지방"),
    ("대전 유성구",    36.3624, 127.3564, 380000, "지방"),
    ("광주 북구",      35.1731, 126.9122, 430000, "지방"),
    ("울산 남구",      35.5384, 129.3114, 320000, "지방"),
    ("강원 춘천시",    37.8813, 127.7298, 280000, "지방"),
]

def random_stores_in_district(d_lat, d_lon, n, spread=0.03):
    """구 중심 좌표 근처에 랜덤 매장 배치"""
    stores = []
    for _ in range(n):
        lat = d_lat + random.uniform(-spread, spread)
        lon = d_lon + random.uniform(-spread, spread)
        stores.append((round(lat, 5), round(lon, 5)))
    return stores

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ── SKT 매장 생성 ──────────────────────────────────────────────────────
skt_stores = []
store_id = 1000

for dist_name, lat, lon, pop, urban_type in DISTRICTS:
    # 인구 규모 + 상권유형에 따라 SKT 매장 수 결정
    base = pop // 100000
    if urban_type == "도심":   n_skt = max(3, base + random.randint(1, 3))
    elif urban_type == "수도권": n_skt = max(2, base + random.randint(0, 2))
    else:                        n_skt = max(1, base + random.randint(-1, 1))

    for (s_lat, s_lon) in random_stores_in_district(lat, lon, n_skt):
        channel_types = ["통합직영", "통합직영", "대리점", "대리점", "판매점"]
        func_types    = ["판매+상담", "판매+상담", "판매+상담", "판매전용", "상담전용"]
        idx = random.randint(0, 4)

        # 월 순증: 도심일수록 높음
        base_adds = {"도심": 45, "수도권": 30, "외곽": 25, "지방": 20}[urban_type]
        monthly_adds = max(1, int(random.gauss(base_adds, base_adds * 0.4)))

        # 평균 LTV (원): 고가요금제 비율 차이
        ltv_base = {"도심": 820000, "수도권": 650000, "외곽": 580000, "지방": 520000}[urban_type]
        avg_ltv = max(200000, int(random.gauss(ltv_base, ltv_base * 0.25)))

        # 월 운영비
        op_cost_base = {"도심": 8500000, "수도권": 6000000, "외곽": 5000000, "지방": 4200000}[urban_type]
        monthly_op_cost = max(2000000, int(random.gauss(op_cost_base, op_cost_base * 0.2)))

        # M/S (%)
        ms_base = {"도심": 35, "수도권": 32, "외곽": 30, "지방": 28}[urban_type]
        ms = round(random.gauss(ms_base, 5), 1)
        ms = max(10, min(60, ms))

        skt_stores.append({
            "store_id": f"SKT-{store_id}",
            "store_name": f"T월드 {dist_name.split()[1]} {store_id % 100:02d}호점",
            "district": dist_name,
            "urban_type": urban_type,
            "lat": s_lat,
            "lon": s_lon,
            "channel_type": channel_types[idx],
            "func_type": func_types[idx],   # 판매+상담 / 판매전용 / 상담전용
            "monthly_adds": monthly_adds,
            "avg_ltv": avg_ltv,
            "monthly_op_cost": monthly_op_cost,
            "ms_pct": ms,
            "pop_served": int(pop / n_skt),
        })
        store_id += 1

# ── 경쟁사 매장 생성 (KT + LGU+) ──────────────────────────────────────
competitor_stores = []
comp_id = 1

for dist_name, lat, lon, pop, urban_type in DISTRICTS:
    base = pop // 100000
    n_kt  = max(1, base + random.randint(-1, 2))
    n_lgu = max(1, base + random.randint(-1, 1))

    for (c_lat, c_lon) in random_stores_in_district(lat, lon, n_kt):
        competitor_stores.append({
            "comp_id": f"KT-{comp_id}",
            "brand": "KT",
            "district": dist_name,
            "lat": c_lat,
            "lon": c_lon,
        })
        comp_id += 1

    for (c_lat, c_lon) in random_stores_in_district(lat, lon, n_lgu):
        competitor_stores.append({
            "comp_id": f"LGU-{comp_id}",
            "brand": "LGU+",
            "district": dist_name,
            "lat": c_lat,
            "lon": c_lon,
        })
        comp_id += 1

# ── 스코어 계산 ────────────────────────────────────────────────────────
RADIUS_KM = 1.5   # 잠식/경쟁 분석 반경

def score_ms_defense(store, all_skt):
    """M/S 방어 점수: 인근 SKT 매장이 많을수록 혼자 폐점해도 M/S 유지 가능 → 점수 높음"""
    nearby = [s for s in all_skt
              if s["store_id"] != store["store_id"]
              and haversine_km(store["lat"], store["lon"], s["lat"], s["lon"]) <= RADIUS_KM]
    n_nearby = len(nearby)
    # 인근 0개=0점, 1개=40점, 2개=65점, 3개=80점, 4개+=100점
    return min(100, n_nearby * 25)

def score_profitability(store, all_skt):
    """수익성 점수: LTV효율 낮을수록 폐점 점수 높음"""
    ltv_efficiency = (store["monthly_adds"] * store["avg_ltv"]) / store["monthly_op_cost"]
    # 전체 평균 대비 얼마나 낮은지 → 낮을수록 폐점 후보
    avg_efficiency = sum(
        (s["monthly_adds"] * s["avg_ltv"]) / s["monthly_op_cost"]
        for s in all_skt
    ) / len(all_skt)
    ratio = ltv_efficiency / avg_efficiency  # 1.0=평균, <1=비효율
    # 비효율일수록 점수 높게 (폐점 후보)
    score = max(0, min(100, int((1 - ratio) * 100 + 50)))
    return score

def score_coverage(store, all_skt, competitors):
    """커버리지 점수: 경쟁사 많고 대체 SKT 있으면 폐점해도 됨 → 점수 높음"""
    nearby_skt  = sum(1 for s in all_skt
                      if s["store_id"] != store["store_id"]
                      and haversine_km(store["lat"], store["lon"], s["lat"], s["lon"]) <= RADIUS_KM)
    nearby_comp = sum(1 for c in competitors
                      if haversine_km(store["lat"], store["lon"], c["lat"], c["lon"]) <= RADIUS_KM)
    # 대체 SKT 있고 (커버리지 보장), 경쟁사 많으면 폐점 부담 낮음
    coverage_ok  = min(50, nearby_skt * 20)
    comp_density = min(50, nearby_comp * 10)
    return coverage_ok + comp_density

def score_cannibalization(store, all_skt):
    """잠식 지수: 인근 SKT 매장 수요 중복 → 높을수록 잠식 중, 폐점 적합"""
    nearby = [s for s in all_skt
              if s["store_id"] != store["store_id"]
              and haversine_km(store["lat"], store["lon"], s["lat"], s["lon"]) <= RADIUS_KM]
    if not nearby:
        return 0
    # 인근 SKT 매장의 평균 순증과 내 순증이 비슷할수록 잠식 중
    avg_nearby_adds = sum(s["monthly_adds"] for s in nearby) / len(nearby)
    similarity = 1 - abs(store["monthly_adds"] - avg_nearby_adds) / max(store["monthly_adds"], avg_nearby_adds, 1)
    return int(similarity * len(nearby) * 20)

def score_channel_function(store):
    """채널 기능 단일성: 단일기능(상담전용/판매전용)이면 폐점 적합"""
    return 80 if store["func_type"] in ("상담전용", "판매전용") else 10

# 가중치
W = {"ms_defense": 0.30, "profitability": 0.30, "coverage": 0.20,
     "cannibalization": 0.12, "channel": 0.08}

results = []
for store in skt_stores:
    s1 = score_ms_defense(store, skt_stores)
    s2 = score_profitability(store, skt_stores)
    s3 = score_coverage(store, skt_stores, competitor_stores)
    s4 = score_cannibalization(store, skt_stores)
    s5 = score_channel_function(store)

    total = (W["ms_defense"]*s1 + W["profitability"]*s2 +
             W["coverage"]*s3 + W["cannibalization"]*s4 + W["channel"]*s5)

    # 판정
    high_signals = sum([s1>=60, s2>=60, s3>=60, s4>=60, s5>=60])
    if high_signals >= 3:   verdict = "🔴 축소 검토"
    elif high_signals >= 1: verdict = "🟡 조건부 유지"
    else:                   verdict = "🟢 유지"

    results.append({
        **store,
        "s1_ms_defense":       s1,
        "s2_profitability":    s2,
        "s3_coverage":         s3,
        "s4_cannibalization":  s4,
        "s5_channel":          s5,
        "closure_score":       round(total, 1),
        "high_signal_count":   high_signals,
        "verdict":             verdict,
    })

results.sort(key=lambda x: x["closure_score"], reverse=True)

# ── 출력 ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  매장수 최적화 스코어 결과  (총 SKT매장: {len(skt_stores)}개 / 경쟁사: {len(competitor_stores)}개)")
print(f"{'='*70}")

red   = [r for r in results if "축소" in r["verdict"]]
yellow= [r for r in results if "조건부" in r["verdict"]]
green = [r for r in results if "유지" in r["verdict"] and "조건부" not in r["verdict"]]

print(f"\n  🔴 축소 검토:  {len(red):3d}개  ({len(red)/len(results)*100:.1f}%)")
print(f"  🟡 조건부 유지: {len(yellow):3d}개  ({len(yellow)/len(results)*100:.1f}%)")
print(f"  🟢 유지:       {len(green):3d}개  ({len(green)/len(results)*100:.1f}%)")

print(f"\n{'─'*70}")
print(f"  폐점 스코어 TOP 15 (가장 적합한 폐점 후보)")
print(f"{'─'*70}")
print(f"  {'매장명':<25} {'상권':<14} {'기능':<8} {'스코어':>6}  {'신호수':>4}  {'판정'}")
print(f"  {'─'*65}")
for r in results[:15]:
    print(f"  {r['store_name']:<25} {r['district']:<14} {r['func_type']:<8} {r['closure_score']:>6.1f}  {r['high_signal_count']:>4}개  {r['verdict']}")

print(f"\n{'─'*70}")
print(f"  유지 권고 TOP 10 (닫으면 안 되는 매장)")
print(f"{'─'*70}")
for r in results[-10:]:
    print(f"  {r['store_name']:<25} {r['district']:<14} {r['func_type']:<8} {r['closure_score']:>6.1f}  {r['high_signal_count']:>4}개  {r['verdict']}")

print(f"\n{'─'*70}")
print(f"  5개 신호별 평균 점수")
print(f"{'─'*70}")
for key, label in [("s1_ms_defense","M/S 방어"), ("s2_profitability","수익성"),
                   ("s3_coverage","커버리지"), ("s4_cannibalization","잠식 지수"),
                   ("s5_channel","채널 기능")]:
    avg = sum(r[key] for r in results) / len(results)
    bar = "█" * int(avg / 5)
    print(f"  {label:<12} {avg:5.1f}점  {bar}")

# JSON 저장
with open("/Users/1112917/mno-store-optimizer/prototype/score_results.json", "w", encoding="utf-8") as f:
    json.dump({"skt_stores": skt_stores, "competitors": competitor_stores, "scores": results}, f, ensure_ascii=False, indent=2)

print(f"\n  ✅ 결과 저장: prototype/score_results.json")
print(f"{'='*70}\n")
