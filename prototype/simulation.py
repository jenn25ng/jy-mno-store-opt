"""
매장수 최적화 시뮬레이션 — 4-Layer Pipeline
Layer 1: DBSCAN (과밀 구역 탐지)
Layer 2: Isolation Forest (이상 패턴 탐지)
Layer 3: Huff 모델 (폐점 영향 시뮬레이션)
Layer 4: 요약 및 신호 생성

※ 매장 위치: SKT/KT/LGU+ 모두 실제 데이터 (성능 데이터는 시뮬레이션)
"""

import json, math, random, os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)

# ─── 실제 매장 데이터 로드 ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
METRO_BBOX = (37.0, 38.0, 126.5, 127.7)  # 수도권 lat/lng 범위

def _in_metro(lat, lng):
    return METRO_BBOX[0] <= lat <= METRO_BBOX[1] and METRO_BBOX[2] <= lng <= METRO_BBOX[3]

def _nearest_zone(lat, lng):
    # haversine은 아래 정의되어 있으나 import 순서 무관 (같은 모듈)
    best, best_dist = COMMERCIAL_ZONES[0], float('inf')
    for z in COMMERCIAL_ZONES:
        dlat = math.radians(z['lat'] - lat)
        dlng = math.radians(z['lng'] - lng)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat)) * math.cos(math.radians(z['lat'])) * math.sin(dlng/2)**2
        d = 6371 * 2 * math.asin(math.sqrt(a))
        if d < best_dist:
            best, best_dist = z, d
    return best, best_dist

def load_real_stores():
    """SKT + KT + LGU+ 실제 데이터 로드. SKT는 성능 데이터를 상권 기반으로 시뮬레이션."""
    stores = []

    # SKT
    skt_path = os.path.join(BASE_DIR, 'skt_stores.json')
    if os.path.exists(skt_path):
        with open(skt_path, encoding='utf-8') as f:
            skt_raw = json.load(f)
        for s in skt_raw:
            if not _in_metro(s['lat'], s['lng']):
                continue
            zone, dist_km = _nearest_zone(s['lat'], s['lng'])
            # 상권 거리에 따른 유동인구 감쇠 (2km 이상이면 저하)
            proximity_factor = max(0.3, 1.0 - dist_km * 0.15)
            foot = zone['foot_traffic'] * proximity_factor

            foot_factor = foot / 100
            monthly_subs = max(5, int(random.gauss(45 * foot_factor, 12)))
            ltv_per_sub = random.gauss(380000, 80000)
            op_cost = random.gauss(8500000, 2000000) * (1 + (1 - foot_factor) * 0.3)
            monthly_revenue = monthly_subs * (ltv_per_sub / 24)
            ms_share = max(0.25, min(0.65, 0.43 + random.gauss(0, 0.04)))

            stores.append({
                'store_id': s['store_id'],
                'carrier': 'SKT',
                'store_name': s['store_name'],
                'address': s.get('address', ''),
                'lat': s['lat'],
                'lng': s['lng'],
                'zone': zone['name'],
                'zone_type': zone['type'],
                'foot_traffic_score': round(foot, 1),
                'monthly_subs': monthly_subs,
                'monthly_profit_krw': round(monthly_revenue - op_cost),
                'ms_share': round(ms_share, 3),
                'op_cost_monthly': round(op_cost),
                'is_real': True,
            })

    # KT
    kt_path = os.path.join(BASE_DIR, 'kt_stores.json')
    if os.path.exists(kt_path):
        with open(kt_path, encoding='utf-8') as f:
            for s in json.load(f):
                if _in_metro(s['lat'], s['lng']):
                    stores.append({
                        'store_id': s['store_id'], 'carrier': 'KT',
                        'store_name': s['store_name'],
                        'lat': s['lat'], 'lng': s['lng'],
                        'zone': '실제데이터', 'zone_type': 'real',
                        'foot_traffic_score': 60,
                        'monthly_subs': None, 'monthly_profit_krw': None,
                        'ms_share': None, 'op_cost_monthly': None, 'is_real': True,
                    })

    # LGU+
    lgu_path = os.path.join(BASE_DIR, 'lgu_stores.json')
    if os.path.exists(lgu_path):
        with open(lgu_path, encoding='utf-8') as f:
            for s in json.load(f):
                if _in_metro(s['lat'], s['lng']):
                    stores.append({
                        'store_id': s['store_id'], 'carrier': 'LGU',
                        'store_name': s['store_name'],
                        'lat': s['lat'], 'lng': s['lng'],
                        'zone': '실제데이터', 'zone_type': 'real',
                        'foot_traffic_score': 60,
                        'monthly_subs': None, 'monthly_profit_krw': None,
                        'ms_share': None, 'op_cost_monthly': None, 'is_real': True,
                    })

    return stores

# ─── 실제 서울/수도권 주요 상권 좌표 ───────────────────────────────────────
COMMERCIAL_ZONES = [
    # 서울 도심
    {"name": "명동", "lat": 37.5635, "lng": 126.9830, "type": "premium", "foot_traffic": 95},
    {"name": "강남역", "lat": 37.4979, "lng": 127.0276, "type": "premium", "foot_traffic": 92},
    {"name": "홍대", "lat": 37.5571, "lng": 126.9245, "type": "premium", "foot_traffic": 88},
    {"name": "신촌", "lat": 37.5596, "lng": 126.9369, "type": "high", "foot_traffic": 75},
    {"name": "건대입구", "lat": 37.5403, "lng": 127.0697, "type": "high", "foot_traffic": 73},
    {"name": "왕십리", "lat": 37.5614, "lng": 127.0387, "type": "high", "foot_traffic": 65},
    {"name": "동대문", "lat": 37.5706, "lng": 127.0098, "type": "high", "foot_traffic": 70},
    {"name": "여의도", "lat": 37.5219, "lng": 126.9244, "type": "premium", "foot_traffic": 80},
    {"name": "잠실", "lat": 37.5132, "lng": 127.1001, "type": "premium", "foot_traffic": 85},
    {"name": "신림", "lat": 37.4843, "lng": 126.9294, "type": "high", "foot_traffic": 68},
    # 수도권
    {"name": "수원역", "lat": 37.2665, "lng": 127.0000, "type": "high", "foot_traffic": 78},
    {"name": "성남분당", "lat": 37.3840, "lng": 127.1223, "type": "high", "foot_traffic": 72},
    {"name": "인천부평", "lat": 37.4908, "lng": 126.7228, "type": "high", "foot_traffic": 70},
    {"name": "안양평촌", "lat": 37.3947, "lng": 126.9527, "type": "mid", "foot_traffic": 60},
    {"name": "고양일산", "lat": 37.6596, "lng": 126.7717, "type": "mid", "foot_traffic": 62},
    {"name": "의정부", "lat": 37.7380, "lng": 127.0473, "type": "mid", "foot_traffic": 55},
    {"name": "남양주", "lat": 37.6359, "lng": 127.2165, "type": "low", "foot_traffic": 42},
    {"name": "용인기흥", "lat": 37.2753, "lng": 127.1145, "type": "mid", "foot_traffic": 58},
    {"name": "파주", "lat": 37.7600, "lng": 126.7097, "type": "low", "foot_traffic": 38},
    {"name": "양주", "lat": 37.7856, "lng": 127.0456, "type": "low", "foot_traffic": 32},
]

def haversine(lat1, lng1, lat2, lng2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def generate_stores():
    stores = load_real_stores()
    skt = [s for s in stores if s['carrier'] == 'SKT']
    kt  = [s for s in stores if s['carrier'] == 'KT']
    lgu = [s for s in stores if s['carrier'] == 'LGU']
    print(f"  실제 데이터 로드: SKT {len(skt)}개, KT {len(kt)}개, LGU+ {len(lgu)}개")
    return stores

# ─── 임계값 설정 (의사결정자가 조정 가능) ────────────────────────────────────
DEFAULT_THRESHOLDS = {
    # 수익성 렌즈: 월 손익 적자 AND Isolation Forest 이상 탐지
    "profitability_max_profit_krw": 0,          # 월 손익 기준 (0 = 적자만)
    "profitability_if_score_max": -0.05,        # anomaly score 상한 (낮을수록 이상)

    # M/S 방어 렌즈: Huff 흡수율 + 인근 SKT 매장 수
    "ms_min_absorption_rate": 0.70,             # 인근 SKT 매장 흡수율 70% 이상
    "ms_min_nearby_skt": 2,                     # 반경 2km 내 SKT 매장 최소 수

    # 커버리지 렌즈: 클러스터 내 매장 수
    "coverage_min_cluster_size": 3,             # 같은 클러스터 내 SKT 매장 3개 이상
}

# ─── LAYER A: DBSCAN 과밀 구역 탐지 ──────────────────────────────────────────
def layer_dbscan(stores):
    skt_stores = [s for s in stores if s["carrier"] == "SKT"]
    coords_rad = np.radians([[s["lat"], s["lng"]] for s in skt_stores])

    eps_rad = 0.8 / 6371.0
    labels = DBSCAN(eps=eps_rad, min_samples=2, algorithm='ball_tree', metric='haversine').fit_predict(coords_rad)

    clusters = {}
    for store, label in zip(skt_stores, labels):
        store["dbscan_cluster"] = int(label)
        store["dbscan_cluster_size"] = 0
        if label != -1:
            clusters.setdefault(label, []).append(store["store_id"])

    for label, members in clusters.items():
        for store in skt_stores:
            if store["dbscan_cluster"] == label:
                store["dbscan_cluster_size"] = len(members)

    overcrowded_clusters = {k: v for k, v in clusters.items() if len(v) >= 3}
    print(f"[DBSCAN] {len(skt_stores)}개 SKT 매장 → {len(overcrowded_clusters)}개 과밀 클러스터")
    return skt_stores, overcrowded_clusters

# ─── LAYER B: Isolation Forest 수익성 이상 탐지 ──────────────────────────────
def layer_isolation_forest(skt_stores):
    valid = [s for s in skt_stores if s["monthly_subs"] is not None]

    features = np.array([
        [s["monthly_subs"], s["monthly_profit_krw"] / 1_000_000,
         s["ms_share"] * 100, s["foot_traffic_score"], s["op_cost_monthly"] / 1_000_000]
        for s in valid
    ])

    X = StandardScaler().fit_transform(features)
    iso = IsolationForest(contamination=0.12, random_state=42, n_estimators=100)
    preds = iso.fit_predict(X)
    scores = iso.score_samples(X)

    for store, pred, score in zip(valid, preds, scores):
        store["if_anomaly"] = bool(pred == -1)
        store["if_score"] = round(float(score), 4)

    print(f"[Isolation Forest] {sum(1 for s in valid if s['if_anomaly'])}/{len(valid)}개 이상 매장 탐지")
    return skt_stores

# ─── LAYER C: Huff 모델 — 전체 SKT 매장 대상 수요 이동 계산 ─────────────────
def layer_huff(skt_stores):
    print(f"[Huff] {len(skt_stores)}개 전체 SKT 매장 시뮬레이션 중...")

    for store in skt_stores:
        nearby_skt = [
            s for s in skt_stores
            if s["store_id"] != store["store_id"]
            and haversine(store["lat"], store["lng"], s["lat"], s["lng"]) < 2.0
        ]
        store["nearby_skt_count"] = len(nearby_skt)

        if not nearby_skt:
            store["huff_absorption_rate"] = 0.0
            store["huff_redistribution"] = []
            store["huff_lost_subs"] = store.get("monthly_subs") or 0
            continue

        weights = []
        for n in nearby_skt:
            dist = max(0.1, haversine(store["lat"], store["lng"], n["lat"], n["lng"]))
            weights.append((n.get("monthly_subs") or 20) / (dist ** 2))

        total_w = sum(weights)
        monthly_subs = store.get("monthly_subs") or 0
        retention = 0.75

        redistribution = []
        for n, w in zip(nearby_skt[:5], weights[:5]):
            ratio = w / total_w if total_w > 0 else 0
            redistribution.append({
                "store_id": n["store_id"],
                "store_name": n["store_name"],
                "absorbed_subs": round(monthly_subs * ratio * retention),
                "distance_km": round(haversine(store["lat"], store["lng"], n["lat"], n["lng"]), 2),
            })

        skt_absorption = sum(w for w in weights) / (total_w or 1) * retention
        store["huff_absorption_rate"] = round(min(skt_absorption, 1.0), 3)
        store["huff_redistribution"] = redistribution
        store["huff_lost_subs"] = round(monthly_subs * (1 - store["huff_absorption_rate"]))

    return skt_stores

# ─── 렌즈 판정: 3개 목적함수 ✓/✗ ────────────────────────────────────────────
def judge_lenses(store, thresholds=None):
    t = thresholds or DEFAULT_THRESHOLDS

    # 수익성: 적자 AND 이상치
    profitability = (
        (store.get("monthly_profit_krw") or 0) < t["profitability_max_profit_krw"]
        and (store.get("if_score") or 0) < t["profitability_if_score_max"]
    )

    # M/S 방어: 인근 SKT가 충분히 흡수 가능
    ms_defense = (
        store.get("huff_absorption_rate", 0) >= t["ms_min_absorption_rate"]
        and store.get("nearby_skt_count", 0) >= t["ms_min_nearby_skt"]
    )

    # 커버리지: 클러스터 내 대체 매장 충분
    coverage = store.get("dbscan_cluster_size", 0) >= t["coverage_min_cluster_size"]

    passed = sum([profitability, ms_defense, coverage])
    if passed == 3:
        verdict = "폐점 유력"
    elif passed == 2:
        verdict = "조건부 검토"
    elif passed == 1:
        verdict = "관찰"
    else:
        verdict = "유지"

    return {
        "lens_profitability": profitability,
        "lens_ms_defense": ms_defense,
        "lens_coverage": coverage,
        "lens_passed": passed,
        "verdict": verdict,
    }

# ─── 최종 집계 ───────────────────────────────────────────────────────────────
def aggregate_results(skt_stores, thresholds=None):
    results = []
    for s in skt_stores:
        lens = judge_lenses(s, thresholds)
        if lens["lens_passed"] == 0:
            continue
        results.append({
            "store_id": s["store_id"],
            "store_name": s["store_name"],
            "zone": s["zone"],
            "lat": s["lat"],
            "lng": s["lng"],
            "monthly_subs": s.get("monthly_subs"),
            "monthly_profit_krw": s.get("monthly_profit_krw"),
            "ms_share": s.get("ms_share"),
            "if_score": s.get("if_score"),
            "dbscan_cluster": s.get("dbscan_cluster"),
            "dbscan_cluster_size": s.get("dbscan_cluster_size"),
            "nearby_skt_count": s.get("nearby_skt_count"),
            "huff_absorption_rate": s.get("huff_absorption_rate"),
            "huff_redistribution": s.get("huff_redistribution"),
            "huff_lost_subs": s.get("huff_lost_subs"),
            **lens,
        })

    results.sort(key=lambda x: (-x["lens_passed"], x.get("monthly_profit_krw") or 0))
    print(f"[판정] 폐점 유력: {sum(1 for r in results if r['verdict']=='폐점 유력')}개 | "
          f"조건부: {sum(1 for r in results if r['verdict']=='조건부 검토')}개 | "
          f"관찰: {sum(1 for r in results if r['verdict']=='관찰')}개")
    return results

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 매장수 최적화 시뮬레이션 시작 ===\n")

    stores = generate_stores()
    skt = [s for s in stores if s["carrier"] == "SKT"]
    kt  = [s for s in stores if s["carrier"] == "KT"]
    lgu = [s for s in stores if s["carrier"] == "LGU"]
    print(f"매장 현황: SKT {len(skt)}개(실제 위치·시뮬레이션 성능), KT {len(kt)}개(실제), LGU+ {len(lgu)}개(실제)\n")

    # 3개 레이어 병렬 실행 (독립적으로 각자 신호 생성)
    skt, overcrowded = layer_dbscan(stores)
    skt = layer_isolation_forest(skt)
    skt = layer_huff(skt)
    print()

    # 렌즈 판정 집계 (임계값 변경 시 여기만 수정)
    results = aggregate_results(skt, thresholds=DEFAULT_THRESHOLDS)

    def default_serializer(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return obj.item()
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    output = {
        "summary": {
            "total_skt": len(skt),
            "total_kt": len(kt),
            "total_lgu": len(lgu),
            "overcrowded_clusters": len(overcrowded),
            "anomaly_stores": sum(1 for s in skt if s.get("if_anomaly")),
            "verdict_counts": {
                "폐점 유력": sum(1 for r in results if r["verdict"] == "폐점 유력"),
                "조건부 검토": sum(1 for r in results if r["verdict"] == "조건부 검토"),
                "관찰": sum(1 for r in results if r["verdict"] == "관찰"),
            },
            "data_note": "SKT/KT/LGU+ 위치 모두 실제 데이터 (2025.06 수집). SKT 성능지표(LTV/손익/M/S)는 상권 기반 시뮬레이션.",
            "thresholds": DEFAULT_THRESHOLDS,
        },
        "all_stores": stores,
        "closure_candidates": results,
        "zones": COMMERCIAL_ZONES,
    }

    with open("prototype/simulation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=default_serializer)

    print(f"\n=== 결과 ===")
    for verdict in ["폐점 유력", "조건부 검토", "관찰"]:
        count = output["summary"]["verdict_counts"][verdict]
        print(f"  {verdict}: {count}개")

    print("\nTOP 폐점 검토 후보:")
    for r in [x for x in results if x["verdict"] == "폐점 유력"][:5]:
        profit_str = f"{r['monthly_profit_krw']//10000:+,}만원" if r["monthly_profit_krw"] else "N/A"
        print(f"  [{r['store_id']}] {r['store_name']}")
        print(f"    손익:{profit_str} | 흡수율:{r['huff_absorption_rate']:.0%} | 클러스터:{r['dbscan_cluster_size']}개")
        print(f"    수익성:{r['lens_profitability']} | M/S방어:{r['lens_ms_defense']} | 커버리지:{r['lens_coverage']}")
        print()

    print(f"결과 저장: prototype/simulation_results.json")
