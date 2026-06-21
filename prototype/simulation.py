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

def _in_korea(lat, lng):
    return 33.0 <= lat <= 38.7 and 124.5 <= lng <= 130.0

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
            if not _in_korea(s['lat'], s['lng']):
                continue
            zone, dist_km = _nearest_zone(s['lat'], s['lng'])
            # 상권 거리에 따른 유동인구 감쇠 (2km 이상이면 저하)
            proximity_factor = max(0.3, 1.0 - dist_km * 0.15)
            foot = zone['foot_traffic'] * proximity_factor

            foot_factor = foot / 100
            # 월 가입 건수: 기본 15건(지역 밀착 수요) + 유동인구 비례 35건 (T월드 평균 20~50건/월)
            monthly_subs = max(10, int(random.gauss(15 + 35 * foot_factor, 10)))
            ltv_per_sub = random.gauss(380_000, 80_000)  # 표시용
            # 월 매출: 신규 가입 1건당 커미션+유지비 기여분 75,000원
            monthly_revenue = monthly_subs * random.gauss(75_000, 18_000)
            # 월 운영비: 임대+인건비 (지역별 차등)
            base_cost = {"premium": 2_800_000, "high": 2_200_000, "mid": 1_600_000, "low": 1_100_000}.get(zone["type"], 1_800_000)
            op_cost = random.gauss(base_cost, base_cost * 0.22) * (1 + (1 - foot_factor) * 0.10)
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
                if _in_korea(s['lat'], s['lng']):
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
                if _in_korea(s['lat'], s['lng']):
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
    {"name": "명동",     "lat": 37.5635, "lng": 126.9830, "type": "premium", "foot_traffic": 95},
    {"name": "강남역",   "lat": 37.4979, "lng": 127.0276, "type": "premium", "foot_traffic": 92},
    {"name": "홍대",     "lat": 37.5571, "lng": 126.9245, "type": "premium", "foot_traffic": 88},
    {"name": "신촌",     "lat": 37.5596, "lng": 126.9369, "type": "high",    "foot_traffic": 75},
    {"name": "건대입구", "lat": 37.5403, "lng": 127.0697, "type": "high",    "foot_traffic": 73},
    {"name": "왕십리",   "lat": 37.5614, "lng": 127.0387, "type": "high",    "foot_traffic": 65},
    {"name": "동대문",   "lat": 37.5706, "lng": 127.0098, "type": "high",    "foot_traffic": 70},
    {"name": "여의도",   "lat": 37.5219, "lng": 126.9244, "type": "premium", "foot_traffic": 80},
    {"name": "잠실",     "lat": 37.5132, "lng": 127.1001, "type": "premium", "foot_traffic": 85},
    {"name": "신림",     "lat": 37.4843, "lng": 126.9294, "type": "high",    "foot_traffic": 68},
    # 수도권
    {"name": "수원역",   "lat": 37.2665, "lng": 127.0000, "type": "high",    "foot_traffic": 78},
    {"name": "성남분당", "lat": 37.3840, "lng": 127.1223, "type": "high",    "foot_traffic": 72},
    {"name": "인천부평", "lat": 37.4908, "lng": 126.7228, "type": "high",    "foot_traffic": 70},
    {"name": "안양평촌", "lat": 37.3947, "lng": 126.9527, "type": "mid",     "foot_traffic": 60},
    {"name": "고양일산", "lat": 37.6596, "lng": 126.7717, "type": "mid",     "foot_traffic": 62},
    {"name": "의정부",   "lat": 37.7380, "lng": 127.0473, "type": "mid",     "foot_traffic": 55},
    {"name": "남양주",   "lat": 37.6359, "lng": 127.2165, "type": "low",     "foot_traffic": 42},
    {"name": "용인기흥", "lat": 37.2753, "lng": 127.1145, "type": "mid",     "foot_traffic": 58},
    {"name": "파주",     "lat": 37.7600, "lng": 126.7097, "type": "low",     "foot_traffic": 38},
    {"name": "양주",     "lat": 37.7856, "lng": 127.0456, "type": "low",     "foot_traffic": 32},
    # 부산
    {"name": "부산서면",   "lat": 35.1575, "lng": 129.0595, "type": "premium", "foot_traffic": 88},
    {"name": "부산남포동", "lat": 35.0979, "lng": 129.0300, "type": "high",    "foot_traffic": 78},
    {"name": "부산해운대", "lat": 35.1631, "lng": 129.1637, "type": "high",    "foot_traffic": 72},
    {"name": "부산동래",   "lat": 35.2058, "lng": 129.0836, "type": "mid",     "foot_traffic": 60},
    {"name": "부산사상",   "lat": 35.1497, "lng": 128.9916, "type": "mid",     "foot_traffic": 55},
    # 대구
    {"name": "대구동성로", "lat": 35.8703, "lng": 128.5942, "type": "premium", "foot_traffic": 85},
    {"name": "대구반월당", "lat": 35.8659, "lng": 128.5951, "type": "high",    "foot_traffic": 75},
    {"name": "대구칠성시장","lat": 35.8856, "lng": 128.5969, "type": "mid",    "foot_traffic": 55},
    # 광주
    {"name": "광주충장로", "lat": 35.1481, "lng": 126.9165, "type": "high",    "foot_traffic": 75},
    {"name": "광주상무지구","lat": 35.1505, "lng": 126.8510, "type": "high",   "foot_traffic": 68},
    # 대전
    {"name": "대전둔산",   "lat": 36.3504, "lng": 127.3845, "type": "high",    "foot_traffic": 72},
    {"name": "대전은행동", "lat": 36.3289, "lng": 127.4278, "type": "mid",     "foot_traffic": 60},
    # 울산
    {"name": "울산삼산",   "lat": 35.5383, "lng": 129.3114, "type": "high",    "foot_traffic": 65},
    {"name": "울산성남동", "lat": 35.5468, "lng": 129.3176, "type": "mid",     "foot_traffic": 55},
    # 경기 외곽
    {"name": "평택",       "lat": 36.9922, "lng": 127.1130, "type": "mid",     "foot_traffic": 55},
    {"name": "천안",       "lat": 36.8151, "lng": 127.1139, "type": "mid",     "foot_traffic": 60},
    {"name": "청주",       "lat": 36.6424, "lng": 127.4890, "type": "mid",     "foot_traffic": 62},
    # 강원
    {"name": "춘천",       "lat": 37.8748, "lng": 127.7341, "type": "mid",     "foot_traffic": 50},
    {"name": "원주",       "lat": 37.3422, "lng": 127.9202, "type": "mid",     "foot_traffic": 52},
    {"name": "강릉",       "lat": 37.7519, "lng": 128.8761, "type": "mid",     "foot_traffic": 48},
    # 전라
    {"name": "전주객사",   "lat": 35.8219, "lng": 127.1489, "type": "mid",     "foot_traffic": 60},
    {"name": "여수",       "lat": 34.7604, "lng": 127.6622, "type": "low",     "foot_traffic": 42},
    {"name": "목포",       "lat": 34.8118, "lng": 126.3922, "type": "low",     "foot_traffic": 40},
    # 경상
    {"name": "창원상남",   "lat": 35.2279, "lng": 128.6810, "type": "mid",     "foot_traffic": 62},
    {"name": "진주",       "lat": 35.1798, "lng": 128.1076, "type": "low",     "foot_traffic": 45},
    {"name": "포항",       "lat": 36.0190, "lng": 129.3435, "type": "mid",     "foot_traffic": 52},
    {"name": "경주",       "lat": 35.8562, "lng": 129.2247, "type": "low",     "foot_traffic": 40},
    {"name": "구미",       "lat": 36.1195, "lng": 128.3446, "type": "mid",     "foot_traffic": 55},
    # 충청
    {"name": "충주",       "lat": 36.9910, "lng": 127.9259, "type": "low",     "foot_traffic": 42},
    {"name": "공주",       "lat": 36.4465, "lng": 127.1190, "type": "low",     "foot_traffic": 38},
    # 제주
    {"name": "제주시청",   "lat": 33.4996, "lng": 126.5312, "type": "mid",     "foot_traffic": 58},
    {"name": "서귀포",     "lat": 33.2541, "lng": 126.5600, "type": "low",     "foot_traffic": 40},
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
    # SKT 시장점유율 ~40% → 3사 경쟁 환경에서 35% 이상이면 '방어 가능' 판정
    "ms_min_absorption_rate": 0.35,             # 인근 SKT 매장 흡수율 35% 이상
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

# ─── LAYER C: Huff 모델 — SKT+KT+LGU+ 경쟁 환경 반영 수요 이동 계산 ─────────
def layer_huff(skt_stores, all_stores):
    """
    Huff 흡수율 = SKT 인근 매장 인력 합 / (SKT + KT + LGU+ 인근 매장 인력 합)
    경쟁사 포함 안 하면 항상 100%가 되어 의미 없음.
    """
    competitor_stores = [s for s in all_stores if s["carrier"] in ("KT", "LGU")]
    print(f"[Huff] {len(skt_stores)}개 SKT 매장 시뮬레이션 (경쟁사 KT {sum(1 for s in all_stores if s['carrier']=='KT')}개 + LGU+ {sum(1 for s in all_stores if s['carrier']=='LGU')}개 포함)...")

    RADIUS_KM = 2.0
    DEFAULT_SUBS = 20  # 경쟁사 규모 추정치 (가입자 데이터 없음)
    retention = 0.75   # 폐점 시 SKT 브랜드 잔존율

    for store in skt_stores:
        lat, lng = store["lat"], store["lng"]

        # 반경 내 인근 SKT 매장
        nearby_skt = [
            s for s in skt_stores
            if s["store_id"] != store["store_id"]
            and haversine(lat, lng, s["lat"], s["lng"]) < RADIUS_KM
        ]
        store["nearby_skt_count"] = len(nearby_skt)

        if not nearby_skt:
            store["huff_absorption_rate"] = 0.0
            store["huff_redistribution"] = []
            store["huff_lost_subs"] = store.get("monthly_subs") or 0
            continue

        def huff_weight(s, default_subs=DEFAULT_SUBS):
            dist = max(0.1, haversine(lat, lng, s["lat"], s["lng"]))
            capacity = s.get("monthly_subs") or default_subs
            return capacity / (dist ** 2)

        # SKT 인력 합
        skt_weights = [huff_weight(n) for n in nearby_skt]
        skt_w = sum(skt_weights)

        # 경쟁사 인력 합 (KT + LGU+, 반경 내)
        nearby_comp = [
            s for s in competitor_stores
            if haversine(lat, lng, s["lat"], s["lng"]) < RADIUS_KM
        ]
        comp_w = sum(huff_weight(c) for c in nearby_comp)

        total_w = skt_w + comp_w
        # SKT 흡수율 = (SKT 인력 / 전체 인력) × 브랜드 잔존율
        skt_absorption = (skt_w / total_w) * retention if total_w > 0 else 0.0

        monthly_subs = store.get("monthly_subs") or 0
        redistribution = []
        for n, w in sorted(zip(nearby_skt, skt_weights), key=lambda x: -x[1])[:5]:
            ratio = w / skt_w if skt_w > 0 else 0
            redistribution.append({
                "store_id": n["store_id"],
                "store_name": n["store_name"],
                "absorbed_subs": round(monthly_subs * ratio * skt_absorption),
                "distance_km": round(haversine(lat, lng, n["lat"], n["lng"]), 2),
            })

        store["huff_absorption_rate"] = round(min(skt_absorption, 1.0), 3)
        store["huff_redistribution"] = redistribution
        store["huff_lost_subs"] = round(monthly_subs * (1 - store["huff_absorption_rate"]))

    return skt_stores

# 도시/지방 zone_type별 임계값 보정 배율
# 도심(premium/high): 매장이 밀집해 있어 더 많은 인근 매장 있어야 폐점 가능
# 지방(low): 자연적으로 매장 수 적음 → 임계값 낮게 설정
ZONE_TYPE_MULTIPLIERS = {
    "premium": {"cluster": 1.5, "nearby": 1.5},
    "high":    {"cluster": 1.25, "nearby": 1.25},
    "mid":     {"cluster": 1.0,  "nearby": 1.0},
    "low":     {"cluster": 0.7,  "nearby": 0.7},
    "real":    {"cluster": 1.0,  "nearby": 1.0},  # 경쟁사 데이터 fallback
}

# ─── 렌즈 판정: 3개 목적함수 ✓/✗ ────────────────────────────────────────────
def judge_lenses(store, thresholds=None):
    t = thresholds or DEFAULT_THRESHOLDS

    # zone_type에 따라 커버리지/인근 임계값 보정
    zone_type = store.get("zone_type", "mid")
    m = ZONE_TYPE_MULTIPLIERS.get(zone_type, ZONE_TYPE_MULTIPLIERS["mid"])
    effective_cluster_min = t["coverage_min_cluster_size"] * m["cluster"]
    effective_nearby_min  = t["ms_min_nearby_skt"] * m["nearby"]

    # 수익성: 적자 AND 이상치
    profitability = (
        (store.get("monthly_profit_krw") or 0) < t["profitability_max_profit_krw"]
        and (store.get("if_score") or 0) < t["profitability_if_score_max"]
    )

    # M/S 방어: 인근 SKT가 충분히 흡수 가능 (zone_type 보정 적용)
    ms_defense = (
        store.get("huff_absorption_rate", 0) >= t["ms_min_absorption_rate"]
        and store.get("nearby_skt_count", 0) >= effective_nearby_min
    )

    # 커버리지: 클러스터 내 대체 매장 충분 (zone_type 보정 적용)
    coverage = store.get("dbscan_cluster_size", 0) >= effective_cluster_min

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
    skt = layer_huff(skt, stores)
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
