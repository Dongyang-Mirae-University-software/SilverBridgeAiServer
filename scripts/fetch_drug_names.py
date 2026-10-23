"""식약처 의약품 제품 허가 목록에서 품목명만 받아 app/data/drug_names.txt로 저장한다.

사용: DATA_GO_KR_KEY(디코딩 키)를 환경변수로 넣고
      python3 -I scripts/fetch_drug_names.py "<목록 오퍼레이션 요청 주소>"
키는 출력하지 않는다. 결과 파일에는 품목명만 한 줄에 하나씩(중복 제거·정렬).
"""
import json, os, sys, time, urllib.parse, urllib.request, urllib.error

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "data", "drug_names.txt")
ROWS = 100


def fetch(endpoint, key, page):
    q = urllib.parse.urlencode({"serviceKey": key, "pageNo": page, "numOfRows": ROWS, "type": "json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{endpoint}?{q}", timeout=30) as r:
                raw = r.read().decode("utf-8")
            data = json.loads(raw)
            err = data.get("OpenAPI_ServiceResponse", {}).get("cmmMsgHeader")
            if err:  # 게이트웨이 오류(키 미등록·한도 초과 등) - 메시지 코드만
                sys.exit(f"API 오류: {err.get('errMsg')} / {err.get('returnAuthMsg')}")
            body = data.get("body") or data.get("response", {}).get("body") or {}
            items = body.get("items") or []
            if isinstance(items, dict):
                items = items.get("item") or []
            if isinstance(items, dict):
                items = [items]
            return int(body.get("totalCount") or 0), items
        except json.JSONDecodeError:
            # XML 오류 응답(키 오류·한도 초과 등) - 키가 섞이지 않게 앞부분 코드만 보여 준다
            sys.exit(f"JSON이 아닌 응답(키·주소 확인 필요): {raw[:160]!r}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  page {page} 재시도 {attempt + 1} ({type(e).__name__})", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    sys.exit(f"page {page} 실패 - 다시 실행하면 처음부터 받는다")


def main():
    if len(sys.argv) != 2:
        sys.exit("사용: python3 -I scripts/fetch_drug_names.py <목록 요청 주소>")
    key = os.environ.get("DATA_GO_KR_KEY", "").strip()
    if not key:
        sys.exit("DATA_GO_KR_KEY 환경변수가 비어 있습니다")
    endpoint = sys.argv[1].split("?")[0]
    total, items = fetch(endpoint, key, 1)
    if not items:
        sys.exit("첫 페이지가 비었습니다 - 주소·키·승인 상태 확인")
    name_field = next((k for k in ("ITEM_NAME", "itemName", "item_name") if k in items[0]), None)
    if not name_field:
        sys.exit(f"품목명 필드를 못 찾았습니다. 필드 목록: {sorted(items[0])}")
    pages = (total + ROWS - 1) // ROWS
    print(f"전체 {total}건, {pages}페이지, 품목명 필드={name_field}")
    names = set()
    for page in range(1, pages + 1):
        if page > 1:
            _, items = fetch(endpoint, key, page)
        names.update((it.get(name_field) or "").strip() for it in items)
        if page % 50 == 0:
            print(f"  {page}/{pages} 페이지, 이름 {len(names)}개")
    names.discard("")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(names)) + "\n")
    print(f"완료: {OUT} ({len(names)}개)")


main()
