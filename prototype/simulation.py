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

# ─── LAYER 1: DBSCAN 과밀 구역 탐지 ─────────────────────────────────────────
def layer1_dbscan(stores):
    skt_stores = [s for s in stores if s["carrier"] == "SKT"]
    coords = np.array([[s["lat"], s["lng"]] for s in skt_stores])
    coords_rad = np.radians(coords)

    # eps=0.8km in radians
    eps_km = 0.8
    eps_rad = eps_km / 6371.0

    db = DBSCAN(eps=eps_rad, min_samples=2, algorithm='ball_tree', metric='haversine')
    labels = db.fit_predict(coords_rad)

    clusters = {}
    for i, (store, label) in enumerate(zip(skt_stores, labels)):
        store["dbscan_cluster"] = int(label)
        store["is_overcrowded"] = label != -1
        if label != -1:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(store["store_id"])

    overcrowded_clusters = {k: v for k, v in clusters.items() if len(v) >= 3}
    print(f"[L1] DBSCAN: {len(skt_stores)} SKT 매장 분석, {len(overcrowded_clusters)}개 과밀 클러스터 발견")
    for cid, members in overcrowded_clusters.items():
        print(f"  클러스터 {cid}: {len(members)}개 매장")
    return skt_stores, overcrowded_clusters

# ─── LAYER 2: Isolation Forest 이상 패턴 탐지 ───────────────────────────────
def layer2_isolation_forest(skt_stores):
    valid = [s for s in skt_stores if s["monthly_subs"] is not None]

    features = np.array([
        [
            s["monthly_subs"],
            s["monthly_profit_krw"] / 1_000_000,
            s["ms_share"] * 100,
            s["foot_traffic_score"],
            s["op_cost_monthly"] / 1_000_000,
        ]
        for s in valid
    ])

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    iso = IsolationForest(contamination=0.12, random_state=42, n_estimators=100)
    preds = iso.fit_predict(X)
    scores = iso.score_samples(X)  # lower = more anomalous

    for store, pred, score in zip(valid, preds, scores):
        store["if_anomaly"] = bool(pred == -1)
        store["if_score"] = round(float(score), 4)

    anomalies = [s for s in valid if s["if_anomaly"]]
    print(f"[L2] Isolation Forest: {len(anomalies)}/{len(valid)} 이상 매장 탐지 (contamination=12%)")
    return skt_stores

# ─── LAYER 3: Huff 모델 폐점 영향 시뮬레이션 ────────────────────────────────
def layer3_huff(skt_stores):
    """과밀+이상 매장 중 폐점 시뮬레이션 — 수요 재배분 계산"""

    candidates = [s for s in skt_stores if s.get("is_overcrowded") and s.get("if_anomaly")]
    if not candidates:
        # 과밀만이라도
        candidates = sorted(
            [s for s in skt_stores if s.get("is_overcrowded")],
            key=lambda x: x.get("if_score", 0)
        )[:5]

    print(f"[L3] Huff 시뮬레이션 대상: {len(candidates)}개 매장")

    for candidate in candidates[:8]:
        nearby = [
            s for s in skt_stores
            if s["store_id"] != candidate["store_id"]
            and haversine(candidate["lat"], candidate["lng"], s["lat"], s["lng"]) < 2.0
        ]

        if not nearby:
            candidate["huff_redistribution"] = None
            continue

        # Huff 확률: 매장규모(순증수) / 거리²
        weights = []
        for n in nearby:
            dist = max(0.1, haversine(candidate["lat"], candidate["lng"], n["lat"], n["lng"]))
            size = n.get("monthly_subs") or 20
            weights.append(size / (dist ** 2))

        total_weight = sum(weights)
        redistributed_subs = candidate.get("monthly_subs") or 0

        redistribution = []
        for n, w in zip(nearby[:5], weights[:5]):
            ratio = w / total_weight if total_weight > 0 else 0
            absorbed = round(redistributed_subs * ratio * 0.75)  # 75% 유지율 가정
            redistribution.append({
                "store_id": n["store_id"],
                "store_name": n["store_name"],
                "absorbed_subs": absorbed,
                "distance_km": round(haversine(candidate["lat"], candidate["lng"], n["lat"], n["lng"]), 2)
            })

        candidate["huff_redistribution"] = redistribution
        candidate["huff_retained_ratio"] = 0.75
        candidate["huff_lost_subs"] = round(redistributed_subs * 0.25)

    return skt_stores, candidates

# ─── LAYER 4: 신호 요약 ──────────────────────────────────────────────────────
def layer4_signals(skt_stores, candidates, overcrowded_clusters):
    results = []

    for s in candidates[:10]:
        signals = []

        if s.get("is_overcrowded"):
            cluster_size = len(overcrowded_clusters.get(s.get("dbscan_cluster", -1), []))
            signals.append(f"과밀 클러스터 (반경 500m 내 {cluster_size}개 SKT 매장)")

        if s.get("if_anomaly"):
            score = s.get("if_score", 0)
            if s.get("monthly_profit_krw", 0) < 0:
                signals.append(f"적자 운영 중 (월 {abs(s['monthly_profit_krw'])//10000}만원 손실)")
            elif s.get("monthly_subs", 0) < 15:
                signals.append(f"저성과 (월 순증 {s['monthly_subs']}건)")

        if s.get("ms_share", 0.43) < 0.35:
            signals.append(f"낮은 M/S ({s['ms_share']*100:.1f}%)")

        huff = s.get("huff_redistribution")
        if huff:
            top = huff[0]
            signals.append(f"폐점 시 {top['distance_km']}km 내 {top['store_name']}이 {top['absorbed_subs']}건 흡수 예상")

        results.append({
            "store_id": s["store_id"],
            "store_name": s["store_name"],
            "zone": s["zone"],
            "lat": s["lat"],
            "lng": s["lng"],
            "monthly_subs": s.get("monthly_subs"),
            "monthly_profit_krw": s.get("monthly_profit_krw"),
            "ms_share": s.get("ms_share"),
            "is_overcrowded": s.get("is_overcrowded"),
            "if_anomaly": s.get("if_anomaly"),
            "if_score": s.get("if_score"),
            "dbscan_cluster": s.get("dbscan_cluster"),
            "huff_redistribution": s.get("huff_redistribution"),
            "huff_lost_subs": s.get("huff_lost_subs"),
            "signals": signals,
            "signal_count": len(signals),
        })

    results.sort(key=lambda x: x["signal_count"], reverse=True)
    return results

# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 매장수 최적화 시뮬레이션 시작 ===\n")

    stores = generate_stores()
    skt = [s for s in stores if s["carrier"] == "SKT"]
    kt  = [s for s in stores if s["carrier"] == "KT"]
    lgu = [s for s in stores if s["carrier"] == "LGU"]
    print(f"매장 현황: SKT {len(skt)}개(실제 위치·시뮬레이션 성능), KT {len(kt)}개(실제), LGU+ {len(lgu)}개(실제)\n")

    skt, overcrowded = layer1_dbscan(stores)
    print()
    skt = layer2_isolation_forest(skt)
    print()
    skt, candidates = layer3_huff(skt)
    print()
    signal_results = layer4_signals(skt, candidates, overcrowded)

    output = {
        "summary": {
            "total_skt": len(skt),
            "total_kt": len(kt),
            "total_lgu": len(lgu),
            "overcrowded_clusters": len(overcrowded),
            "anomaly_stores": sum(1 for s in skt if s.get("if_anomaly")),
            "closure_candidates": len(signal_results),
            "data_note": "SKT/KT/LGU+ 위치 모두 실제 데이터 (2025.06 수집). SKT 성능지표(LTV/손익/M/S)는 상권 기반 시뮬레이션.",
        },
        "all_stores": stores,
        "closure_candidates": signal_results,
        "zones": COMMERCIAL_ZONES,
    }

    def default_serializer(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return obj.item()
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open("prototype/simulation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=default_serializer)

    print(f"\n=== 결과 ===")
    print(f"과밀 클러스터: {len(overcrowded)}개")
    print(f"이상 탐지 매장: {output['summary']['anomaly_stores']}개")
    print(f"복수 신호 폐점 후보: {len(signal_results)}개\n")

    print("TOP 폐점 검토 후보:")
    for r in signal_results[:5]:
        profit_str = f"{r['monthly_profit_krw']//10000:+,}만원" if r['monthly_profit_krw'] else "N/A"
        print(f"  [{r['store_id']}] {r['store_name']}")
        print(f"    월순증:{r['monthly_subs']}건 | 손익:{profit_str} | 신호:{r['signal_count']}개")
        for sig in r["signals"]:
            print(f"    • {sig}")
        print()

    print(f"결과 저장: prototype/simulation_results.json")
