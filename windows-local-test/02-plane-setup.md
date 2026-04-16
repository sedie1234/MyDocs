# Phase 2: Plane 구축

WSL2 터미널에서 진행합니다.

## 1. 저장소 클론

```bash
cd ~/workspace/plane
git clone https://github.com/makeplane/plane.git .
```

## 2. 환경 파일 생성

```bash
cp .env.example .env
cp apps/web/.env.example apps/web/.env
cp apps/api/.env.example apps/api/.env
cp apps/space/.env.example apps/space/.env
cp apps/admin/.env.example apps/admin/.env
cp apps/live/.env.example apps/live/.env
```

## 3. 실행

```bash
docker compose up -d
```

프록시 포함 전체 서비스가 올라갑니다. 최초 실행 시 이미지 빌드에 10~20분 소요.

## 4. 상태 확인

```bash
docker compose ps
```

모든 컨테이너가 `Up` 상태인지 확인.

## 5. 접속

| URL | 용도 |
|-----|------|
| http://localhost/god-mode/ | 관리자 등록 (최초 1회) |
| http://localhost | 메인 웹 앱 |

### 최초 설정
1. http://localhost/god-mode/ → 관리자 계정 등록
2. http://localhost → 로그인 → 워크스페이스 생성

## 6. 기능 테스트 체크리스트

- [ ] 워크스페이스 생성
- [ ] 프로젝트 생성
- [ ] Work Items (이슈) 생성/편집/삭제
- [ ] 이슈에 파일 첨부
- [ ] Cycles (스프린트) 생성 및 이슈 할당
- [ ] Modules (모듈) 생성 및 이슈 그룹화
- [ ] Views (커스텀 뷰) 필터 생성/저장
- [ ] Pages (문서) 작성
- [ ] 멤버 초대 및 권한 설정
- [ ] Analytics (분석) 대시보드 확인

## 관리 명령어

```bash
# 종료
docker compose down

# 재시작 (데이터 유지)
docker compose up -d

# 로그 확인
docker compose logs -f api
docker compose logs -f web

# 완전 초기화 (데이터 삭제)
docker compose down -v
```
