# Phase 3: GitLab 구축

WSL2 터미널에서 진행합니다.
Plane과 포트가 충돌하지 않도록 GitLab은 8880/8443/2222 포트를 사용합니다.

## 1. 작업 디렉토리 생성

```bash
cd ~/workspace/gitlab
mkdir -p config logs data
```

## 2. docker-compose.yml 생성

```bash
cat > docker-compose.yml << 'EOF'
services:
  gitlab:
    image: gitlab/gitlab-ce:latest
    container_name: gitlab
    restart: always
    hostname: localhost
    environment:
      GITLAB_OMNIBUS_CONFIG: |
        external_url 'http://localhost:8880'
        gitlab_rails['gitlab_shell_ssh_port'] = 2222
        nginx['listen_port'] = 8880
        nginx['listen_https'] = false
    ports:
      - "8880:8880"
      - "8443:443"
      - "2222:22"
    volumes:
      - ./config:/etc/gitlab
      - ./logs:/var/log/gitlab
      - ./data:/var/opt/gitlab
    shm_size: '256m'
EOF
```

## 3. 실행

```bash
docker compose up -d
```

> GitLab 최초 실행 시 내부 초기화에 3~5분 소요됩니다.

## 4. 상태 확인

```bash
# 컨테이너 상태
docker compose ps

# 초기화 진행 확인 (healthy가 될 때까지 대기)
docker inspect --format='{{.State.Health.Status}}' gitlab
```

## 5. 초기 관리자 비밀번호 확인

```bash
docker exec -it gitlab grep 'Password:' /etc/gitlab/initial_root_password
```

> 이 파일은 24시간 후 자동 삭제됩니다. 비밀번호를 기록해두세요.

## 6. 접속

| URL | 용도 |
|-----|------|
| http://localhost:8880 | GitLab 웹 UI |

- ID: `root`
- PW: 위에서 확인한 초기 비밀번호

로그인 후 비밀번호 변경 권장.

## 7. 기능 테스트 체크리스트

- [ ] 로그인 및 비밀번호 변경
- [ ] 사용자 생성
- [ ] 프로젝트(저장소) 생성
- [ ] 코드 push/pull (HTTP)
- [ ] 코드 push/pull (SSH, 포트 2222)
- [ ] Merge Request 생성/리뷰/병합
- [ ] CI/CD 파이프라인 실행
- [ ] Issue 생성 및 관리

### Git SSH 설정 (선택)

```bash
# SSH 키 생성
ssh-keygen -t ed25519

# 공개키 복사 후 GitLab → Settings → SSH Keys에 등록
cat ~/.ssh/id_ed25519.pub

# SSH config 설정
cat >> ~/.ssh/config << 'EOF'
Host localhost
  Port 2222
  IdentityFile ~/.ssh/id_ed25519
EOF
```

## 관리 명령어

```bash
# 종료
docker compose down

# 재시작
docker compose up -d

# 로그
docker compose logs -f gitlab

# 완전 초기화
docker compose down -v
```
