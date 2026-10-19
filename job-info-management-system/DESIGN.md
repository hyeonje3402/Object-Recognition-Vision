# 취업 정보 관리 시스템 (Professional Recruitment System) 디자인 가이드라인

본 문서는 Stitch MCP에서 추출한 '취업 정보 관리 시스템'의 디자인 시스템 규칙(색상, 타이포그래피, 레이아웃 등)을 정리한 가이드라인입니다.

---

## 1. 브랜드 및 스타일 개요 (Brand & Style)

본 디자인 시스템은 높은 신뢰도와 효율성을 요구하는 채용 관리 서비스에 최적화되어 있습니다.
*   **브랜드 아이덴티티:** 전문성(Professional), 신뢰성(Reliable), 신속한 의사결정(Decisive)
*   **디자인 테마:** 모던 코퍼레이트 (Modern Corporate)
*   **핵심 가치:**
    *   **명확성과 정밀성(Clarity and Precision):** 모든 시각적 요소는 명확한 기능적 목적을 가집니다.
    *   **신뢰 기반 미학(Trust-Based Aesthetics):** 정갈한 네이비 블루 톤과 철저한 그리드 정렬을 통해 높은 신뢰감을 제공합니다.
    *   **데이터 효율성(Data Efficiency):** 이력서, 채용 공고, 지원 상태 등 방대한 정보를 눈의 피로 없이 빠르게 탐색할 수 있습니다.
    *   **정제된 인터랙션(Refined Interactivity):** 부드러운 전환 효과와 고도 변화를 활용하여 채용 퍼널 전반에서 사용자의 행동을 자연스럽게 유도합니다.

---

## 2. 색상 규격 (Colors)

브랜드의 주조색은 신뢰를 주는 **네이비 블루**이며, 강조색(CTA)은 사용자의 인터랙션을 유도하는 **비브란트 블루**입니다.

### 2.1. 핵심 색상 팔레트 (Core Palette)
*   **Primary (주조색):** `#002045` (네비게이션, 헤더, 브랜딩 영역)
*   **Primary Container:** `#1A365D`
*   **On Primary:** `#FFFFFF`
*   **Secondary (보조/강조색):** `#0061A5` 또는 `#3182CE` (주요 실행 버튼(CTA), 선택 상태, 진행 표시 등)
*   **Secondary Container:** `#66AFFE`
*   **On Secondary:** `#FFFFFF`
*   **Background (배경색):** `#F8F9FF` (깨끗하고 넓은 느낌을 주는 연한 블루 계열)
*   **On Background:** `#0B1C30`

### 2.2. 표면 및 외곽선 색상 (Surface & Outline)
*   **Surface:** `#F8F9FF`
*   **Surface Dim:** `#CBDBF5`
*   **Surface Bright:** `#F8F9FF`
*   **Surface Container (Lowest):** `#FFFFFF` (카드 및 리스트 내부 콘텐츠 배경)
*   **Surface Container (Low):** `#EFF4FF`
*   **Surface Container:** `#E5EEFF`
*   **Surface Container (High):** `#DCE9FF`
*   **Surface Container (Highest):** `#D3E4FE`
*   **On Surface:** `#0B1C30`
*   **On Surface Variant:** `#43474E`
*   **Outline:** `#74777F`
*   **Outline Variant:** `#C4C6CF`

### 2.3. 시스템 색상 (Semantic Colors)
상태값 표시(승인, 대기, 거절 등)에 쓰이는 고대비 의미론적 색상입니다.
*   **Error (경고/실패):** `#BA1A1A`
*   **On Error:** `#FFFFFF`
*   **Error Container:** `#FFDAD6`
*   **On Error Container:** `#93000A`

---

## 3. 타이포그래피 (Typography)

서체는 국문과 영문 모두에서 뛰어난 가독성을 자랑하는 **Inter**를 기본 서체로 채택하였습니다. 정보 전달 중심의 복잡한 데이터 구조에서 텍스트 계층 구조를 명확히 구분해 줍니다.

| 스타일명 | 용도 | Font Family | Size | Weight | Line Height | Letter Spacing |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **headline-lg** | 데스크톱 메인 타이틀 | Inter | 32px | 700 (Bold) | 40px | -0.02em |
| **headline-lg-mobile** | 모바일 메인 타이틀 | Inter | 24px | 700 (Bold) | 32px | -0.02em |
| **headline-md** | 섹션 타이틀, 카드 헤더 | Inter | 20px | 600 (Semi-Bold)| 28px | -0.01em |
| **body-lg** | 본문 텍스트 (큰 글씨) | Inter | 16px | 400 (Regular) | 24px | - |
| **body-md** | 기본 본문, 정보 설명 | Inter | 14px | 400 (Regular) | 20px | - |
| **label-md** | 폼 레이블, 캡션 | Inter | 12px | 600 (Semi-Bold)| 16px | 0.02em |
| **status-badge** | 상태 배지 텍스트 | Inter | 11px | 700 (Bold) | 14px | 0.05em |

---

## 4. 레이아웃 및 간격 규칙 (Layout & Spacing)

**8px 선형 스케일**을 기반으로 시각적 듬직함과 규칙성 있는 간격을 설정합니다.

*   **기본 스케일 (Spacing Scale):**
    *   `xs`: 4px
    *   `sm`: 8px
    *   `md`: 16px
    *   `lg`: 24px
    *   `xl`: 32px
*   **그리드 시스템 (Grid):**
    *   데스크톱: 12열 가변 그리드 (Fluid Grid)
    *   모바일: 1열 가변 레이아웃 (Fluid Layout)
*   **모바일 여백:** 모바일 화면에서는 사이드 여백을 `16px`로 통일하여 작은 화면에서도 정보 전달 영역을 극대화합니다.
*   **카드 여백:** 리스트 내 카드 컴포넌트는 내부 여백(Padding) `16px` 및 카드 간 위아래 간격(Margin) `12px`를 유지합니다.
*   **터치 타겟 영역:** 모든 인터랙션 요소(버튼, 링크 등)는 모바일 및 터치 환경을 위해 시각적 크기와 무관하게 최소 **44px × 44px** 이상의 터치 영역을 확보해야 합니다.

---

## 5. 모서리 둥글기 및 고도화 (Shapes & Elevation)

### 5.1. 모서리 곡률 (Rounded Corners)
기본적으로 신뢰감 있고 부드러운 곡률을 제공합니다.
*   **Small (`sm`):** `0.25rem (4px)`
*   **Default:** `0.5rem (8px)` - 카드, 인풋, 버튼 등 대다수 컴포넌트의 표준값
*   **Medium (`md`):** `0.75rem (12px)`
*   **Large (`lg`):** `1rem (16px)`
*   **Extra Large (`xl`):** `1.5rem (24px)` - 상태 배지나 알림창 등에 적용
*   **Full (`full`):** `9999px` - 완전한 원형 배지 및 알약 모양

### 5.2. 고도 및 그림자 (Elevation & Depth)
빛의 산란을 사실적으로 표현한 부드럽고 얇은 그림자를 활용하여 계층을 분리합니다.
*   **Surface 0 (기본 배경):** `#F8F9FF` 위에 요소를 배치합니다.
*   **Surface 1 (콘텐츠 영역):** 카드나 화이트 박스 형태는 `#FFFFFF`를 적용하고 입체감을 주기 위해 소프트 그림자(`0px 2px 4px rgba(0, 0, 0, 0.05)`)를 부여합니다.
*   **인터랙션 피드백:** 요소에 호버(Hover)하거나 포커스할 때 그림자의 반경과 깊이를 자연스럽게 높여 클릭 가능함을 직관적으로 유도합니다.

---

## 6. 주요 컴포넌트 설계 규칙 (Component Rules)

### 6.1. 버튼 (Buttons)
*   **Primary Button (주요 행동):** 배경색 `#3182CE` (또는 `#0061A5`) + 텍스트 흰색. 8px 곡률. "지원하기", "공고 등록" 등에 최우선 적용.
*   **Secondary Button (보조 행동):** 아웃라인 스타일 (1px `#1A365D` 테두리) 또는 흐린 네이비 텍스트. "저장하기", "목록으로" 등에 적용.
*   **버튼 높이:** 모바일 주요 액션은 `48px`, 데스크톱 리스트 내 일반 액션은 `40px`로 가이드라인을 세웁니다.

### 6.2. 상태 표시 배지 (Status Badges)
배경색은 각 상태 의미론적 색상의 불투명도 **10%**로 처리하고, 텍스트는 해당 색상의 불투명도 **100%**로 처리하여 최적의 가독성과 접근성을 제공합니다.
*   **승인 (APPROVED):** 초록색 톤 배경 및 텍스트
*   **대기 (PENDING):** 주황색/황색 톤 배경 및 텍스트
*   **거절 (REJECTED):** 빨간색 톤 배경 및 텍스트

### 6.3. 입력 필드 (Input Fields)
*   **기본 상태:** 배경 `#FFFFFF`, 테두리 `1px #E2E8F0`, 곡률 `8px`.
*   **포커스 상태:** 테두리 `2px #3182CE` 및 연한 파란색 그림자 효과.
*   **레이블 배치:** 입력 폼 바로 위에 `label-md` 폰트 스타일로 텍스트를 배치하여 가독성을 높이고 빠른 작성 처리를 유도합니다.
