const { test, expect } = require('@playwright/test');

test.describe('Network Interception & API Testing', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('성공적인 API 응답 모킹(Mocking) 및 테이블 렌더링 검증', async ({ page }) => {
    const fetchBtn = page.locator('#fetch-users-btn');
    const tableRows = page.locator('#users-tbody tr');

    // 1. /api/users API 호출 인터셉트 설정
    await page.route('**/api/users', async (route) => {
      const mockUsers = [
        { id: 1, name: '심청', email: 'cheong@hyo.org' },
        { id: 2, name: '임꺽정', email: 'kkukjung@hero.net' },
        { id: 3, name: '장길산', email: 'gilsan@rebel.com' }
      ];
      
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockUsers)
      });
    });

    // 2. Fetch Users 클릭
    await fetchBtn.click();

    // 3. 테이블 데이터 검증 (Row 개수가 3개여야 함)
    await expect(tableRows).toHaveCount(3);
    
    // 첫 번째 행 값 검증
    const firstRowText = await tableRows.nth(0).innerText();
    expect(firstRowText).toContain('심청');
    expect(firstRowText).toContain('cheong@hyo.org');

    // 세 번째 행 값 검증
    const lastRowText = await tableRows.nth(2).innerText();
    expect(lastRowText).toContain('장길산');
    expect(lastRowText).toContain('gilsan@rebel.com');
  });

  test('API 호출 실패(500 Error) 모킹 및 화면 예외 처리 검증', async ({ page }) => {
    const fetchBtn = page.locator('#fetch-users-btn');
    const tableRows = page.locator('#users-tbody tr');

    // 1. /api/users API 호출 시 500 Internal Server Error 발생하도록 모킹
    await page.route('**/api/users', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Database Connection Fail' })
      });
    });

    // 2. Fetch Users 클릭
    await fetchBtn.click();

    // 3. 화면에 에러 메시지가 표기되는지 검증
    await expect(tableRows).toHaveCount(1);
    await expect(tableRows).toContainText('데이터 조회 실패');
    await expect(tableRows).toContainText('HTTP error! status: 500');
  });

  test('Playwright APIRequestContext를 이용한 직접 API 호출 테스트', async ({ request }) => {
    // UI 브라우저를 띄우지 않고 직접 HTTP 요청을 수행하여 서버 상태 검증
    // 로컬 서버의 index.html을 직접 get
    const response = await request.get('/index.html');
    
    // HTTP status가 200대(OK)인지 검증
    expect(response.ok()).toBeTruthy();
    expect(response.status()).toBe(200);

    // 응답 텍스트에 특정 스트링이 포함되어 있는지 검증
    const text = await response.text();
    expect(text).toContain('Playwright Sandbox Playground');
  });
});
