const { test, expect } = require('@playwright/test');

test.describe('Visual Regression & Screenshot Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('페이지 전체 및 특정 카드 엘리먼트 스크린샷 캡처', async ({ page }) => {
    // 1. 전체 페이지 스크린샷 저장 (유저 디버깅용 파일 생성 예제)
    await page.screenshot({ path: 'tests/screenshots/entire-page.png' });

    // 2. 특정 카드 요소(#form-section) 스크린샷 저장
    const formCard = page.locator('#form-section');
    await formCard.screenshot({ path: 'tests/screenshots/form-card.png' });
  });

  test('라이트 모드 및 다크 모드 비주얼 회귀 테스트', async ({ page }) => {
    // [라이트 모드 비주얼 회귀 비교]
    // maxDiffPixels와 threshold 옵션을 지정하여 크로스 플랫폼 간 렌더링 엔진의 미세한 차이를 허용
    await expect(page).toHaveScreenshot('homepage-light-mode.png', {
      maxDiffPixels: 200,
      threshold: 0.2,
      animations: 'disabled' // 애니메이션 중지
    });

    // [다크 모드로 전환 후 비주얼 회귀 비교]
    const themeBtn = page.locator('#theme-toggle-btn');
    await themeBtn.click();

    // 다크 모드 활성화 확인 (body 클래스 검증)
    await expect(page.locator('body')).toHaveClass(/dark-mode/);

    await expect(page).toHaveScreenshot('homepage-dark-mode.png', {
      maxDiffPixels: 200,
      threshold: 0.2,
      animations: 'disabled'
    });
  });
});
