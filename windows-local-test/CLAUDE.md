# Windows 로컬 테스트 환경 - Claude Code 설정

이 파일을 Windows WSL2의 작업 디렉토리에 복사하여 사용합니다.

## 프로젝트 개요
Plane(프로젝트 관리) + GitLab(소스코드 관리)을 사내 서버 배포 전 로컬에서 테스트하는 환경.

## 디렉토리 구조
```
~/workspace/
├── plane/          ← Plane (docker compose로 실행)
└── gitlab/         ← GitLab (docker compose로 실행)
```

## 주요 명령어

### Plane
```bash
cd ~/workspace/plane
docker compose up -d          # 시작
docker compose down           # 종료
docker compose logs -f api    # API 로그
```

### GitLab
```bash
cd ~/workspace/gitlab
docker compose up -d          # 시작
docker compose down           # 종료
docker compose logs -f gitlab # 로그
```

### 상태 확인
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

## 포트 구성
- Plane: 80 (프록시 경유, localhost)
- GitLab: 8880 (HTTP), 8443 (HTTPS), 2222 (SSH)

## 목표
- 기능 테스트 및 운영 설정값 확정
- 확정된 설정을 사내 서버(172.20.4.60)에 이관
