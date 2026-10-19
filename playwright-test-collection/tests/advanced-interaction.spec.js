const { test, expect } = require('@playwright/test');

test.describe('Advanced Interactions Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('마우스 드래그 앤 드롭 테스트', async ({ page }) => {
    const source = page.locator('#drag-source');
    const target = page.locator('#drop-target');
    const status = page.locator('#drag-status');

    // 1. dragTo API를 사용하여 드래그 앤 드롭 수행
    await source.dragTo(target);

    // 2. 결과 검증
    await expect(target).toHaveClass(/dropped/);
    await expect(target).toHaveText('Success Dropped!');
    await expect(status).toHaveText('드래그 앤 드롭 성공!');
  });

  test('파일 업로드 테스트 (setInputFiles)', async ({ page }) => {
    const fileInput = page.locator('#file-input');
    const fileNameText = page.locator('#file-name');
    const uploadStatus = page.locator('#upload-status');

    // 1. 메모리 상에서 가상의 파일 데이터 생성 및 업로드
    await fileInput.setInputFiles({
      name: 'playwright-test.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('Playwright E2E Test File Upload content')
    });

    // 2. 결과 검증 (선택된 파일명 및 성공 문구 노출)
    await expect(fileNameText).toHaveText('playwright-test.txt');
    await expect(uploadStatus).toBeVisible();
    await expect(uploadStatus).toHaveText(/성공: playwright-test.txt/);
  });

  test('Iframe 내부 요소 접근 및 입력 테스트', async ({ page }) => {
    // 1. frameLocator를 통해 iframe 영역 획득
    const frame = page.frameLocator('#test-iframe');
    const iframeInput = frame.locator('#iframe-input');
    const iframeSubmitBtn = frame.locator('#iframe-submit-btn');
    const iframeStatus = frame.locator('#iframe-status');

    // 2. iframe 내부 요소 입력 및 버튼 클릭
    await iframeInput.fill('Hello Iframe!');
    await iframeSubmitBtn.click();

    // 3. iframe 내부 상태 텍스트 검증
    await expect(iframeStatus).toHaveText('제출 완료: "Hello Iframe!"');
  });

  test('새로운 팝업 윈도우 감지 및 팝업 제어 테스트', async ({ context, page }) => {
    const openPopupBtn = page.locator('#open-popup-btn');
    const popupStatus = page.locator('#popup-status');

    // 1. 팝업 발생 이벤트를 기다리는 Promise 생성
    const popupPromise = page.waitForEvent('popup');

    // 2. 팝업을 유발하는 버튼 클릭
    await openPopupBtn.click();

    // 3. 팝업 페이지 객체 획득
    const popup = await popupPromise;

    // 4. 팝업 페이지의 URL 및 타이틀 확인
    await popup.waitForLoadState();
    expect(popup.url()).toContain('iframe.html?type=popup');
    await expect(popup).toHaveTitle('Iframe Content');

    // 5. 팝업 페이지 내부 요소(예: 팝업 알림 텍스트) 존재 확인
    const popupNotice = popup.locator('#popup-notice');
    await expect(popupNotice).toBeVisible();
    await expect(popupNotice).toContainText('이 페이지는 독립된 팝업 창에서 실행 중입니다.');

    // 6. 팝업 창 내 폼 입력 테스트
    await popup.locator('#iframe-input').fill('Text inside Popup');
    await popup.locator('#iframe-submit-btn').click();
    await expect(popup.locator('#iframe-status')).toHaveText('제출 완료: "Text inside Popup"');

    // 7. 팝업 창 닫기 및 메인 페이지 상태 전이 확인
    await popup.close();
    await expect(popupStatus).toHaveText('팝업 창이 닫혔습니다.');
  });
});
