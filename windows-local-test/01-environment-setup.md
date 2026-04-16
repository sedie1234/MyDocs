# Phase 1: 환경 준비

## 1. WSL2 설치

PowerShell (관리자 권한)에서 실행:

```powershell
wsl --install
```

재부팅 후 Ubuntu가 자동 설치됨. 사용자명/비밀번호 설정.

이미 WSL이 있으면 버전 확인:
```powershell
wsl --version
wsl --list --verbose
```

## 2. Docker Desktop 설치

1. https://www.docker.com/products/docker-desktop/ 에서 다운로드
2. 설치 시 "Use WSL 2 based engine" 체크
3. 설치 후 재부팅
4. Docker Desktop → Settings → Resources → WSL Integration → Ubuntu 활성화

WSL2 터미널에서 확인:
```bash
docker --version
docker compose version
```

## 3. WSL2 메모리 설정

Plane + GitLab 동시 실행을 위해 메모리 할당 필요.

Windows에서 `%USERPROFILE%\.wslconfig` 파일 생성:

```ini
[wsl2]
memory=12GB
swap=4GB
processors=4
```

적용:
```powershell
wsl --shutdown
```

## 4. Claude Code 설치 (WSL2 내부)

WSL2 Ubuntu 터미널에서:

```bash
# Node.js 설치 (nvm 사용)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc
nvm install --lts

# Claude Code 설치
npm install -g @anthropic-ai/claude-code

# 실행
cd ~/workspace
claude
```

## 5. 작업 디렉토리 생성

```bash
mkdir -p ~/workspace/plane
mkdir -p ~/workspace/gitlab
```

## 환경 준비 완료 체크리스트

- [ ] WSL2 Ubuntu 실행 확인
- [ ] Docker Desktop 실행 + WSL2 연동 확인
- [ ] `docker compose version` 출력 확인
- [ ] WSL2 메모리 12GB 할당 확인
- [ ] Claude Code 설치 및 실행 확인
