const { test, expect } = require('@playwright/test');

test.describe('Basic Form & E2E Tests', () => {
  // 각 테스트 전 메인 페이지 방문
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('타이틀 및 필수 요소 렌더링 확인', async ({ page }) => {
    // 1. 페이지 타이틀 확인
    await expect(page).toHaveTitle(/Playwright Sandbox App/);

    // 2. 메인 헤더 제목 검증
    const headerTitle = page.locator('.header-container h1');
    await expect(headerTitle).toBeVisible();
    await expect(headerTitle).toHaveText('Playwright Sandbox Playground');
  });

  test('빈 폼 제출 시 유효성 검사 경고 메시지 노출 검증', async ({ page }) => {
    const submitBtn = page.locator('#submit-btn');
    const nameError = page.locator('#name-error');
    const emailError = page.locator('#email-error');

    // 초기 상태에서 에러 메시지는 숨겨져 있어야 함
    await expect(nameError).not.toBeVisible();
    await expect(emailError).not.toBeVisible();

    // 폼 제출
    await submitBtn.click();

    // 에러 메시지가 화면에 노출되어야 함
    await expect(nameError).toBeVisible();
    await expect(nameError).toHaveText('이름은 필수 항목입니다.');
    await expect(emailError).toBeVisible();
    await expect(emailError).toHaveText('올바른 이메일 주소를 입력하세요.');
  });

  test('폼 작성 및 정상 제출 시 성공 결과 출력 검증', async ({ page }) => {
    const nameInput = page.locator('#user-name');
    const emailInput = page.locator('#user-email');
    const subscribeCheckbox = page.locator('#subscribe-checkbox');
    const browserSelect = page.locator('#preferred-browser');
    const submitBtn = page.locator('#submit-btn');
    const resultBox = page.locator('#form-result');
    const resultMessage = page.locator('#result-message');

    // 폼 필드 입력
    await nameInput.fill('홍길동');
    await emailInput.fill('gildong@example.com');
    
    // 체크박스 클릭 및 상태 확인
    await subscribeCheckbox.check();
    expect(await subscribeCheckbox.isChecked()).toBe(true);

    // 드롭다운 셀렉트 선택
    await browserSelect.selectOption('webkit');
    expect(await browserSelect.inputValue()).toBe('webkit');

    // 제출
    await submitBtn.click();

    // 결과 박스 가시성 확인 및 출력 메시지 검증
    await expect(resultBox).toBeVisible();
    await expect(resultMessage).toContainText('홍길동');
    await expect(resultMessage).toContainText('gildong@example.com');
    await expect(resultMessage).toContainText('예'); // 구독여부
    await expect(resultMessage).toContainText('webkit'); // 브라우저
  });
});
