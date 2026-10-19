# Playwright Test Collection Guide

이 가이드는 Playwright를 활용해 구현된 종합 테스트 컬렉션의 유형별 시나리오, 검증 방법 및 사용된 API 목록을 정리한 문서입니다. 
본 프로젝트는 테스트의 일관성과 재현성을 보장하기 위해, 내장된 **로컬 모의 애플리케이션(`app/` 디렉토리)**을 대상으로 테스트를 구동합니다.

---

## 📂 프로젝트 구조 및 테스트 구성

```text
playwright-test-collection/
├── app/                        # 테스트 대상이 되는 로컬 웹 애플리케이션
│   ├── index.html              # 메인 플레이그라운드 페이지
│   ├── style.css               # 스타일링 (라이트/다크모드 지원)
│   ├── app.js                  # 인터랙티브 동작 스크립트
│   └── iframe.html             # iframe 및 팝업용 서브 페이지
├── tests/                      # Playwright 테스트 수트
│   ├── basic-form.spec.js      # 기본 E2E 및 Form 검증
│   ├── advanced-interaction.spec.js # 드래그앤드롭, 파일업로드, iframe, 팝업
│   ├── network-mocking.spec.js # API 모킹 및 API Client 직접 요청
│   ├── visual-regression.spec.js # 스크린샷 캡처 및 비주얼 회귀 테스트
│   └── emulation-monitoring.spec.js # GPS, 콘솔 로그 감지, 오프라인 모드
├── package.json                # 프로젝트 의존성 및 스크립트
└── playwright.config.js        # Playwright 전역 설정 파일
```

---

## 🏷️ 1. 테스트 컬렉션 유형별 상세 시나리오

### 유형 A. Basic E2E & Form Validation
* **대상 테스트 파일**: [`tests/basic-form.spec.js`](file:///C:/Users/82107/.gemini/antigravity-ide/scratch/playwright-test-collection/tests/basic-form.spec.js)
* **설명**: 웹사이트 방문 후 UI 구성 요소들이 화면에 정상적으로 로드되었는지 확인하고, 폼 유효성 검사 경고 기능 및 성공적인 폼 제출 결과를 검증합니다.

| 시나리오 명 | 테스트 흐름 (Steps) | 검증 요소 (Assertions) |
| :--- | :--- | :--- |
| **타이틀 및 필수 요소 렌더링 확인** | 1. 홈(`/`) 페이지 방문<br>2. 문서 타이틀 및 헤더의 텍스트 로케이터 추출 | - `page` 타이틀이 "Playwright Sandbox App"인지 검증<br>- 헤더가 화면에 보이고 텍스트가 정확한지 검증 |
| **빈 폼 제출 시 유효성 검증** | 1. 홈 페이지 방문<br>2. 폼 내 빈 상태로 '제출하기' 버튼 클릭 | - 이름 필수 경고 및 이메일 형식 경고 문구가 즉각 표시(visible)되는지 확인 |
| **폼 작성 및 정상 제출 성공** | 1. 홈 페이지 방문<br>2. 이름과 이메일 값 입력<br>3. '뉴스레터 구독' 체크박스 체크<br>4. 브라우저 선택 셀렉트에서 'WebKit' 지정<br>5. '제출하기' 클릭 | - 결과 박스가 화면에 노출됨을 확인<br>- 입력한 정보(이름, 이메일, 구독여부(예), webkit)가 결과 박스 텍스트에 포함되었는지 검증 |

* **사용된 핵심 API**: `goto()`, `locator()`, `fill()`, `check()`, `selectOption()`, `click()`, `toHaveTitle()`, `toBeVisible()`, `toHaveText()`, `toContainText()`

---

### 유형 B. Advanced User Interactions
* **대상 테스트 파일**: [`tests/advanced-interaction.spec.js`](file:///C:/Users/82107/.gemini/antigravity-ide/scratch/playwright-test-collection/tests/advanced-interaction.spec.js)
* **설명**: 드래그 앤 드롭, 로컬 파일 탐색기를 통한 파일 업로드, Iframe 안쪽에 중첩된 DOM 요소 조작, 멀티 탭 및 브라우저 팝업의 생성부터 종료까지의 이벤트를 다룹니다.

| 시나리오 명 | 테스트 흐름 (Steps) | 검증 요소 (Assertions) |
| :--- | :--- | :--- |
| **마우스 드래그 앤 드롭** | 1. 드래그 대상 요소와 드롭 존 요소를 지정<br>2. 드롭 존으로 요소를 끌어서 놓기 | - 드롭 존의 CSS 클래스에 `.dropped`가 추가되었는지 검증<br>- 내부 텍스트가 "Success Dropped!"로 갱신되었는지 검증 |
| **파일 업로드 (setInputFiles)** | 1. 파일 input 요소를 획득<br>2. 가상 텍스트 파일 버퍼를 주입하여 업로드 수행 | - 파일명 텍스트에 파일명이 정확히 명시되었는지 검증<br>- 성공 배너가 가시화되었는지 검증 |
| **Iframe 내부 요소 접근 및 제어** | 1. `frameLocator`를 통해 `#test-iframe` 스코프 지정<br>2. 프레임 내부 input과 submit 요소를 타겟팅하여 입력 및 클릭 | - iframe 내부의 상태 표시줄 텍스트가 입력했던 값으로 정상 변경되었는지 검증 |
| **새 팝업 탭 생성 및 독립 제어** | 1. `waitForEvent('popup')` 이벤트 대기 시작<br>2. 팝업 실행 버튼 클릭<br>3. 새롭게 생성된 팝업 페이지의 로드 대기 | - 팝업 URL에 쿼리 스트링 파라미터가 포함되었는지 확인<br>- 팝업 내 전용 공지사항 메시지가 노출되는지 확인<br>- 팝업 내 폼을 조작 후 팝업을 닫았을 때, 부모 페이지에 "팝업 창이 닫혔습니다." 문구가 반영되는지 검증 |

* **사용된 핵심 API**: `dragTo()`, `setInputFiles()`, `frameLocator()`, `waitForEvent('popup')`, `popup.url()`, `popup.close()`

---

### 유형 C. Network Mocking & Direct API Calls
* **대상 테스트 파일**: [`tests/network-mocking.spec.js`](file:///C:/Users/82107/.gemini/antigravity-ide/scratch/playwright-test-collection/tests/network-mocking.spec.js)
* **설명**: 프론트엔드의 fetch 네트워크 호출을 중간에 가로채서(Mocking) 가짜 데이터를 제공하거나, 고의로 500 에러 상태 코드를 주입하여 화면 예외 처리가 작동하는지 검사합니다. 또한 브라우저를 띄우지 않고 API 요청을 독자적으로 테스트합니다.

| 시나리오 명 | 테스트 흐름 (Steps) | 검증 요소 (Assertions) |
| :--- | :--- | :--- |
| **성공 API 응답 모킹 & 테이블 렌더링** | 1. `page.route`로 `/api/users` 경로 매칭<br>2. JSON 형태의 3명 유저 목록 응답 주입<br>3. UI에서 'Fetch Users' 클릭 | - UI 테이블 내부 행(tr) 개수가 mock 데이터 개수인 3개인지 확인<br>- 렌더링된 각 셀에 mock 유저 정보가 정상 표시되었는지 검증 |
| **API 호출 실패 500 에러 모킹** | 1. `page.route`로 `/api/users` 경로에 status 500 응답 주입<br>2. 'Fetch Users' 클릭 | - UI 테이블 내부에 실패 안내 행(tr)이 그려지는지 검증<br>- 500 에러 문구(`HTTP error! status: 500`)가 올바르게 인쇄되었는지 검증 |
| **직접 API 호출 검증** | 1. Playwright의 `request` Context를 활용<br>2. `/index.html` 리소스에 직접 GET 요청 전송 | - 응답 코드가 `200` 및 `ok` 상태인지 확인<br>- 응답 바디 텍스트에 HTML 시그니처 텍스트가 담겨있는지 확인 |

* **사용된 핵심 API**: `route()`, `route.fulfill()`, `request.get()`, `response.ok()`, `response.status()`, `response.text()`

---

### 유형 D. Visual Regression & Screenshots
* **대상 테스트 파일**: [`tests/visual-regression.spec.js`](file:///C:/Users/82107/.gemini/antigravity-ide/scratch/playwright-test-collection/tests/visual-regression.spec.js)
* **설명**: 마이너한 디자인 깨짐이나 레이아웃 변경을 방지하기 위해 스크린샷 이미지 픽셀 단위 비교 검증(Visual Regression)을 수행합니다.

| 시나리오 명 | 테스트 흐름 (Steps) | 검증 요소 (Assertions) |
| :--- | :--- | :--- |
| **전체 및 요소 스크린샷 캡처** | 1. 전체 뷰포트 스크린샷 저장<br>2. `#form-section`에 해당하는 로케이터 영역만 크롭(crop)하여 저장 | - 지정한 로컬 디렉토리 경로(`tests/screenshots/`)에 실제 이미지 파일이 정상 생성되었는지 확인 |
| **라이트/다크모드 비주얼 회귀 테스트** | 1. 라이트 모드 상태에서 스크린샷 스냅샷 비교<br>2. 테마 토글 버튼 클릭하여 다크 모드 활성화<br>3. 다크 모드 상태에서 스냅샷 비교 | - `toHaveScreenshot` API를 활용하여 기존 레퍼런스 이미지와 현재 렌더링 픽셀이 일치하는지 대조 검사<br>- 플랫폼별 폰트 렌더링 등의 오차 방지를 위해 `maxDiffPixels` 및 `threshold` 설정 구성 |

* **사용된 핵심 API**: `page.screenshot()`, `locator.screenshot()`, `expect(page).toHaveScreenshot()`

---

### 유형 E. Browser Emulation & Realtime Monitoring
* **대상 테스트 파일**: [`tests/emulation-monitoring.spec.js`](file:///C:/Users/82107/.gemini/antigravity-ide/scratch/playwright-test-collection/tests/emulation-monitoring.spec.js)
* **설명**: GPS 인위적 위치값 전달, 브라우저가 출력하는 console 메시지와 예기치 못한 Unhandled JS Error 모니터링, 그리고 네트워크 오프라인 에뮬레이션 상태에서의 동작 신뢰성을 테스트합니다.

| 시나리오 명 | 테스트 흐름 (Steps) | 검증 요소 (Assertions) |
| :--- | :--- | :--- |
| **위치 정보(Geolocation) 에뮬레이션** | 1. `browser.newContext`로 Geolocation 값을 서울시청 좌표로 모킹하고 권한 수락 설정 부여<br>2. 페이지 내 '내 위치 정보 확인' 버튼 클릭 | - 페이지 결과창에 모킹한 위도(37.5665)와 경도(126.9780) 텍스트가 정확히 도출되는지 확인 |
| **콘솔 로그 및 JS 에러 모니터링** | 1. `page.on('console')` 및 `page.on('pageerror')` 리스너 작동<br>2. 콘솔 로그 유발 버튼 및 런타임 에러 유발 버튼 순차 클릭 | - 이벤트 리스너를 통해 수집된 로그에 사용자 클릭 메시지가 담겼는지 검증<br>- `SandboxRuntimeException` 예외 에러가 정상 포착되었는지 감지 |
| **네트워크 오프라인 모드 검증** | 1. `context.setOffline(true)`로 인터넷 연결을 단절시킴<br>2. API 요청 유발 버튼 클릭 | - 요청 실패가 로컬 화면 테이블에 `데이터 조회 실패` 상태 문구로 무너지지 않고 사용자에게 우아하게 안내되는지 검증 |

* **사용된 핵심 API**: `browser.newContext({ geolocation, permissions })`, `page.on('console')`, `page.on('pageerror')`, `context.setOffline()`

---

## 📊 2. Playwright 핵심 API Map 요약

테스트 컬렉션에 골고루 반영된 주요 Playwright API의 기능과 목적을 한데 묶어 제공합니다.

| API 분류 | API 메서드 명칭 | 주요 설명 및 용도 |
| :--- | :--- | :--- |
| **Navigation** | `page.goto(url)` | 테스트 타겟 URL로 브라우저를 이동시킵니다. |
| **Selectors** | `page.locator(selector)` | CSS Selector, Text, ID 등을 기반으로 DOM 요소를 지정합니다. |
| **User Inputs** | `locator.fill(value)` | 입력 폼 필드(Text, Email 등)에 문자열을 기입합니다. |
| | `locator.check()` | 체크박스 또는 라디오 버튼을 선택 상태로 전환합니다. |
| | `locator.selectOption(value)` | `<select>` 드롭다운 메뉴에서 지정한 옵션을 선택합니다. |
| | `locator.click()` | 버튼, 링크 등의 요소를 클릭 이벤트로 작동시킵니다. |
| **Advanced Actions** | `locator.dragTo(target)` | 마우스로 소스 요소를 잡아 대상 타겟 영역으로 끌어다 놓습니다. |
| | `locator.setInputFiles(files)` | `<input type="file">` 요소에 직접 로컬 파일을 첨부합니다. |
| | `page.frameLocator(selector)` | 페이지 내 `<iframe>` 내부 영역의 DOM 노드를 탐색하기 위한 진입점을 제공합니다. |
| | `page.waitForEvent('popup')` | `window.open` 등으로 유발되는 새로운 탭/윈도우의 생성을 감지하고 대기합니다. |
| **Mocking / API** | `page.route(url, handler)` | 특정 네트워크 요청 경로를 인터셉트하여 대기시킵니다. |
| | `route.fulfill(options)` | 가로챈 네트워크 요청에 가짜 JSON이나 커스텀 상태 코드(e.g., 500)를 응답으로 보냅니다. |
| | `request.get(url)` | 브라우저 렌더링 없이 APIRequestContext를 사용해 서버에 직접 REST API를 요청합니다. |
| **Visual Validation**| `page.screenshot()` | 현재 브라우저 창의 전체 모습을 PNG 스크린샷 이미지 파일로 남깁니다. |
| | `expect(page).toHaveScreenshot()`| 기존의 스냅샷 레퍼런스 이미지 파일과 현재 화면 픽셀을 대조하여 회귀 여부를 식별합니다. |
| **Emulation / Obs** | `browser.newContext(options)`| Geolocation, 사용자 권한(permissions), 디바이스 뷰포트 크기 등을 모킹한 격리 컨텍스트를 선언합니다. |
| | `page.on('console')` | 브라우저 개발자 도구 콘솔에 남는 로그 메시지를 실시간으로 모니터링 및 인터셉트합니다. |
| | `page.on('pageerror')` | 웹 페이지 내에서 발생했으나 처리되지 않은 JS Unhandled Exception을 캐치합니다. |
| | `context.setOffline(state)` | 네트워크 온라인/오프라인 상태를 강제 스위칭하여 브라우저의 오프라인 대응력을 검사합니다. |

---

## 🚀 3. 테스트 실행 가이드

1. **활성 워크스페이스 열기**
   * 에디터 혹은 터미널에서 다음 디렉토리를 활성 워크스페이스로 설정하세요:
     `C:\Users\82107\.gemini\antigravity-ide\scratch\playwright-test-collection`

2. **의존성 설치**
   * 터미널을 열고 필요한 패키지 및 브라우저를 다운로드합니다.
     ```bash
     npm install
     npx playwright install
     ```

3. **테스트 구동**
   * **기본 터미널 실행 (헤드리스)**:
     ```bash
     npx playwright test
     ```
   * **대화형 UI 모드로 실행 (테스트 분석 및 타임라인 확인용)**:
     ```bash
     npx playwright test --ui
     ```
   * **디버그 모드로 한 단계씩 실행**:
     ```bash
     npx playwright test --debug
     ```
