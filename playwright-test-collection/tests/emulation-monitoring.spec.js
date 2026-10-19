const { test, expect } = require('@playwright/test');

test.describe('Emulation & Monitoring Tests', () => {
  
  test('위치 정보(Geolocation) 권한 부여 및 좌표 에뮬레이션 검증', async ({ browser }) => {
    // 1. 서울시청의 좌표(37.5665, 126.9780)와 권한을 사전 정의한 컨텍스트 생성
    const context = await browser.newContext({
      geolocation: { latitude: 37.5665, longitude: 126.9780 },
      permissions: ['geolocation']
    });
    
    const page = await context.newPage();
    await page.goto('/');

    const geoBtn = page.locator('#geolocation-btn');
    const geoStatus = page.locator('#geolocation-status');

    // 2. 위치 가져오기 버튼 클릭
    await geoBtn.click();

    // 3. 에뮬레이션된 위도와 경도가 표시되는지 검증
    await expect(geoStatus).toHaveText('위도: 37.5665, 경도: 126.9780');
    
    await context.close();
  });

  test('브라우저 콘솔 로그 감지 및 JS Unhandled Exception 모니터링', async ({ page }) => {
    const logs = [];
    const pageErrors = [];

    // 1. console 이벤트 리스너 등록
    page.on('console', (msg) => {
      logs.push(msg.text());
    });

    // 2. pageerror(런타임 에러) 이벤트 리스너 등록
    page.on('pageerror', (err) => {
      pageErrors.push(err.message);
    });

    await page.goto('/');

    // 3. 콘솔 로그 유발 버튼 클릭
    await page.locator('#trigger-log-btn').click();
    
    // 로그 감지 확인 (배열에 저장되었는지 검사)
    expect(logs.some(log => log.includes('[Playwright Sandbox] Custom action log clicked'))).toBe(true);

    // 4. 런타임 에러 유발 버튼 클릭
    await page.locator('#trigger-error-btn').click();

    // 비동기 에러 발생 대기 (50ms 디레이 감안)
    await page.waitForTimeout(100);

    // 런타임 에러 감지 확인
    expect(pageErrors.length).toBeGreaterThan(0);
    expect(pageErrors[0]).toContain('SandboxRuntimeException');
  });

  test('네트워크 오프라인(Offline) 모드 에뮬레이션 및 예외 처리 검증', async ({ context, page }) => {
    await page.goto('/');

    // 1. 네트워크를 오프라인 상태로 강제 전이
    await context.setOffline(true);

    const fetchBtn = page.locator('#fetch-users-btn');
    const tableRows = page.locator('#users-tbody tr');

    // 2. Fetch Users 클릭 (오프라인이므로 fetch 호출 실패가 예상됨)
    await fetchBtn.click();

    // 3. 오프라인에 따른 브라우저 에러(Failed to fetch) 노출 확인
    await expect(tableRows).toHaveCount(1);
    await expect(tableRows).toContainText('데이터 조회 실패');
    
    // 오프라인 복구 (리소스 반환 전 원복)
    await context.setOffline(false);
  });
});
