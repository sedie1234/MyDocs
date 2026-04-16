# Windows 로컬 테스트 환경 구축 계획

## 배경

- 사내 서버(172.20.4.60)의 포트포워딩 승인 대기 중
- 승인 전까지 Windows 로컬에서 Plane + GitLab을 먼저 테스트
- 기능 파악 및 운영 설정값 확정 후, 서버에 동일 구성 적용 예정

## 환경 선택: WSL2 권장

| 항목 | WSL2 | Windows 네이티브 |
|------|------|-------------------|
| Docker 호환성 | Linux 컨테이너 그대로 실행 | Docker Desktop 필요 (WSL2 백엔드 사용) |
| 명령어 호환 | 서버(Ubuntu)와 동일 | PowerShell 별도 문법 |
| 설정 파일 재사용 | 서버 설정 그대로 복사 가능 | 경로 등 수정 필요 |
| 서버 이관 | 스크립트/설정 그대로 이관 | 변환 작업 필요 |
| Claude Code 사용 | CLI 직접 사용 가능 | WSL 통해 사용 |

**결론: WSL2 + Docker Desktop 조합을 사용**
- 서버와 동일한 Ubuntu 환경이므로 설정/스크립트 재사용 가능
- 테스트 후 서버 이관 시 변환 작업 불필요

## 구축 순서

### Phase 1: 환경 준비
1. WSL2 설치 및 Ubuntu 배포판 설정
2. Docker Desktop 설치 (WSL2 백엔드 활성화)
3. Claude Code 설치 (WSL2 내부)

### Phase 2: Plane 구축
1. Plane 저장소 클론
2. docker compose로 전체 서비스 실행
3. 관리자 계정 등록 및 기능 테스트

### Phase 3: GitLab 구축
1. GitLab Docker 이미지 실행
2. 초기 설정 및 관리자 계정 설정
3. 프로젝트 생성/push/pull 테스트

### Phase 4: 서버 이관
1. 포트포워딩 승인 확인
2. 서버에 동일 설정 적용 (현재 서버에 이미 Plane 구축 완료)
3. GitLab 서버 설치
4. 내부망 접속 테스트

## 파일 구성

```
docs/windows-local-test/
├── README.md                 ← 이 문서 (전체 계획)
├── 01-environment-setup.md   ← WSL2 + Docker Desktop 설치 가이드
├── 02-plane-setup.md         ← Plane 구축 가이드
├── 03-gitlab-setup.md        ← GitLab 구축 가이드
├── 04-server-migration.md    ← 서버 이관 체크리스트
└── CLAUDE.md                 ← Claude Code 프로젝트 설정
```

## 시스템 요구사항

- Windows 10 (2004 이상) 또는 Windows 11
- RAM: 16GB 이상 권장 (Plane + GitLab 동시 실행)
- 디스크: 40GB 이상 여유 공간
- CPU: 4코어 이상
