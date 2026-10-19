// 1. Dark Mode Theme Toggle
const themeToggleBtn = document.getElementById('theme-toggle-btn');
const themeIcon = themeToggleBtn.querySelector('.icon');

// LocalStorage check
if (localStorage.getItem('theme') === 'dark') {
  document.body.classList.add('dark-mode');
  themeIcon.textContent = '☀️';
  themeToggleBtn.innerHTML = '<span class="icon">☀️</span> Light Mode';
}

themeToggleBtn.addEventListener('click', () => {
  document.body.classList.toggle('dark-mode');
  const isDark = document.body.classList.contains('dark-mode');
  
  if (isDark) {
    localStorage.setItem('theme', 'dark');
    themeToggleBtn.innerHTML = '<span class="icon">☀️</span> Light Mode';
    console.log('Theme changed to Dark Mode');
  } else {
    localStorage.setItem('theme', 'light');
    themeToggleBtn.innerHTML = '<span class="icon">🌙</span> Dark Mode';
    console.log('Theme changed to Light Mode');
  }
});

// 2. Basic Form Validation & Submission
const feedbackForm = document.getElementById('feedback-form');
const userNameInput = document.getElementById('user-name');
const userEmailInput = document.getElementById('user-email');
const subscribeCheckbox = document.getElementById('subscribe-checkbox');
const preferredBrowserSelect = document.getElementById('preferred-browser');
const formResult = document.getElementById('form-result');
const resultMessage = document.getElementById('result-message');

feedbackForm.addEventListener('submit', (e) => {
  e.preventDefault();
  
  let isValid = true;
  
  // Name validate
  if (!userNameInput.value.trim()) {
    document.getElementById('name-error').classList.add('visible');
    isValid = false;
  } else {
    document.getElementById('name-error').classList.remove('visible');
  }
  
  // Email validate
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(userEmailInput.value.trim())) {
    document.getElementById('email-error').classList.add('visible');
    isValid = false;
  } else {
    document.getElementById('email-error').classList.remove('visible');
  }
  
  if (isValid) {
    const name = userNameInput.value.trim();
    const email = userEmailInput.value.trim();
    const isSubscribed = subscribeCheckbox.checked ? '예' : '아니오';
    const browser = preferredBrowserSelect.value || '선택 안 함';
    
    resultMessage.innerHTML = `
      <strong>이름:</strong> ${name}<br>
      <strong>이메일:</strong> ${email}<br>
      <strong>구독 여부:</strong> ${isSubscribed}<br>
      <strong>브라우저:</strong> ${browser}
    `;
    
    formResult.classList.remove('hidden');
    console.log('Form submitted successfully', { name, email, isSubscribed, browser });
  }
});

// 3. Drag & Drop
const dragSource = document.getElementById('drag-source');
const dropTarget = document.getElementById('drop-target');
const dragStatus = document.getElementById('drag-status');

dragSource.addEventListener('dragstart', (e) => {
  e.dataTransfer.setData('text/plain', 'drag-element');
  dragSource.style.opacity = '0.5';
  dragStatus.textContent = '드래그 중입니다...';
});

dragSource.addEventListener('dragend', () => {
  dragSource.style.opacity = '1';
});

dropTarget.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropTarget.classList.add('hover');
});

dropTarget.addEventListener('dragleave', () => {
  dropTarget.classList.remove('hover');
});

dropTarget.addEventListener('drop', (e) => {
  e.preventDefault();
  dropTarget.classList.remove('hover');
  
  const data = e.dataTransfer.getData('text/plain');
  if (data === 'drag-element') {
    dropTarget.classList.add('dropped');
    dropTarget.textContent = 'Success Dropped!';
    dragStatus.textContent = '드래그 앤 드롭 성공!';
    console.log('Drag and Drop succeeded');
  }
});

// 4. File Upload
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
const uploadStatus = document.getElementById('upload-status');

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    const file = e.target.files[0];
    fileName.textContent = file.name;
    uploadStatus.textContent = `성공: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    uploadStatus.classList.remove('hidden');
    uploadStatus.classList.add('success');
    console.log('File selected:', file.name);
  } else {
    fileName.textContent = '선택된 파일 없음';
    uploadStatus.classList.add('hidden');
  }
});

// 5. Popup Open
const openPopupBtn = document.getElementById('open-popup-btn');
const popupStatus = document.getElementById('popup-status');

openPopupBtn.addEventListener('click', () => {
  popupStatus.textContent = '팝업 창을 열었습니다.';
  const popup = window.open('iframe.html?type=popup', 'PlaywrightPopup', 'width=500,height=400');
  
  if (popup) {
    console.log('Popup window opened');
    
    // Check closure
    const timer = setInterval(() => {
      if (popup.closed) {
        clearInterval(timer);
        popupStatus.textContent = '팝업 창이 닫혔습니다.';
        console.log('Popup window closed');
      }
    }, 1000);
  } else {
    popupStatus.textContent = '팝업 창이 차단되었습니다.';
    console.warn('Popup blocked');
  }
});

// 6. Network Fetch Users
const fetchUsersBtn = document.getElementById('fetch-users-btn');
const usersTbody = document.getElementById('users-tbody');

fetchUsersBtn.addEventListener('click', async () => {
  fetchUsersBtn.disabled = true;
  fetchUsersBtn.textContent = 'Loading...';
  
  try {
    const response = await fetch('/api/users');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Clear existing body
    usersTbody.innerHTML = '';
    
    if (data && data.length > 0) {
      data.forEach(user => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${user.id}</td>
          <td>${user.name}</td>
          <td>${user.email}</td>
        `;
        usersTbody.appendChild(tr);
      });
      console.log('Fetched users data:', data);
    } else {
      usersTbody.innerHTML = '<tr><td colspan="3" class="text-center">표시할 데이터가 없습니다.</td></tr>';
    }
  } catch (error) {
    console.error('Fetch error:', error);
    usersTbody.innerHTML = `<tr><td colspan="3" class="text-center" style="color: var(--danger-color)">데이터 조회 실패: ${error.message}</td></tr>`;
  } finally {
    fetchUsersBtn.disabled = false;
    fetchUsersBtn.textContent = 'Fetch Users';
  }
});

// 7. Geolocation
const geolocationBtn = document.getElementById('geolocation-btn');
const geolocationStatus = document.getElementById('geolocation-status');

geolocationBtn.addEventListener('click', () => {
  geolocationStatus.textContent = '위치 조회 중...';
  
  if (!navigator.geolocation) {
    geolocationStatus.textContent = 'Geolocation이 지원되지 않는 브라우저입니다.';
    return;
  }
  
  navigator.geolocation.getCurrentPosition(
    (position) => {
      const { latitude, longitude } = position.coords;
      geolocationStatus.textContent = `위도: ${latitude.toFixed(4)}, 경도: ${longitude.toFixed(4)}`;
      console.log('Geolocation success:', { latitude, longitude });
    },
    (error) => {
      geolocationStatus.textContent = `위치 조회 실패: ${error.message}`;
      console.error('Geolocation error:', error);
    }
  );
});

// 8. Console & Error Trigger
const triggerLogBtn = document.getElementById('trigger-log-btn');
const triggerErrorBtn = document.getElementById('trigger-error-btn');

triggerLogBtn.addEventListener('click', () => {
  console.log('[Playwright Sandbox] Custom action log clicked at ' + new Date().toLocaleTimeString());
});

triggerErrorBtn.addEventListener('click', () => {
  console.error('[Playwright Sandbox] Custom error generated for monitoring');
  // Trigger unhandled rejection or runtime error
  setTimeout(() => {
    throw new Error('SandboxRuntimeException: 테스트 실패 감지용 에러입니다.');
  }, 50);
});
