# flowsight → GitHub push (당신 Mac 터미널에서 1회)

> repo 루트 = `flowsight/` (자가완결 패키지 + experiments + notebook). 시크릿 없음 → **Public** 권장(Colab `git clone`이 가장 단순).

```bash
cd ~/Desktop/magi/flowsight
git init -b main
git add .
git commit -m "FlowSight v0 scaffold: geometry/physics/sim/perception + experiments"
```

**A) GitHub CLI(`gh`)가 있으면 한 줄:**
```bash
gh repo create flowsight --public --source=. --push
```

**B) 없으면 수동:** github.com/new → 이름 `flowsight`, Public, 템플릿 없음 → 생성 후:
```bash
git remote add origin https://github.com/<당신아이디>/flowsight.git
git push -u origin main
```

끝나면 **repo URL**을 알려주세요. 그다음부터 Colab(clone→GPU→설치→실데이터 지도데모)은 제가 브라우저로 돌립니다.

- Private로 하려면 Colab clone에 토큰이 필요해 한 단계 늘어납니다 → Public 권장.
- `git`이 없다면: `xcode-select --install` 후 위 명령 실행.
