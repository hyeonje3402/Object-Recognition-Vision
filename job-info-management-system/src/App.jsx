import { useState } from 'react'
import './App.css'

function App() {
  const [email, setEmail] = useState(() => {
    const savedEmail = localStorage.getItem('rememberedEmail')
    const savedRememberMe = localStorage.getItem('rememberMe') === 'true'
    return savedRememberMe && savedEmail ? savedEmail : ''
  })
  const [password, setPassword] = useState('')
  const [rememberMe, setRememberMe] = useState(() => {
    return localStorage.getItem('rememberMe') === 'true'
  })
  const [showPassword, setShowPassword] = useState(false)
  
  // Interactive UI states
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [emailFocus, setEmailFocus] = useState(false)
  const [pwdFocus, setPwdFocus] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const [successMessage, setSuccessMessage] = useState('')

  const handleLogin = async (e) => {
    e.preventDefault()
    setErrorMessage('')
    setSuccessMessage('')

    if (!email) {
      setErrorMessage('이메일 주소를 입력해 주세요.')
      return
    }
    if (!password) {
      setErrorMessage('비밀번호를 입력해 주세요.')
      return
    }

    // Basic email validation regex
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email)) {
      setErrorMessage('유효한 이메일 형식이 아닙니다.')
      return
    }

    if (password.length < 4) {
      setErrorMessage('비밀번호는 최소 4자 이상이어야 합니다.')
      return
    }

    setIsSubmitting(true)
    const backendUrl = (import.meta.env.REACT_APP_BACKEND || '').replace(/\/$/, '')
    try {
      const response = await fetch(`${backendUrl}/api/v1/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      })


      const result = await response.json()
      if (response.ok && result.status === 'success') {
        setSuccessMessage('로그인에 성공했습니다!')
        
        // Store JWT token
        if (result.data) {
          localStorage.setItem('token', result.data)
        }

        // Remember Me functionality
        if (rememberMe) {
          localStorage.setItem('rememberedEmail', email)
          localStorage.setItem('rememberMe', 'true')
        } else {
          localStorage.removeItem('rememberedEmail')
          localStorage.removeItem('rememberMe')
        }

        console.log('Login successful, token stored:', result.data)
      } else {
        setErrorMessage(result.message || '로그인에 실패했습니다. 이메일과 비밀번호를 확인해 주세요.')
      }
    } catch (error) {
      console.error('Login error:', error)
      setErrorMessage('서버와 통신하는 중 오류가 발생했습니다.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleCancel = () => {
    setEmail('')
    setPassword('')
    setErrorMessage('')
    setSuccessMessage('')
  }

  const handleSocialLogin = (platform) => {
    setSuccessMessage(`${platform} 간편 로그인을 실행합니다.`)
  }

  return (
    <div className="login-page-container">
      {/* Left Decorative Panel (Desktop Only) */}
      <div className="branding-panel">
        <div className="branding-glow-1"></div>
        <div className="branding-glow-2"></div>
        
        <div className="branding-content">
          <h1 className="branding-title">
            가장 앞선 인재 채용의 시작, <br />
            RecruitMaster
          </h1>
          <p className="branding-desc">
            지능형 매칭 알고리즘과 투명한 데이터 대시보드로 채용 프로세스의 신뢰도를 높이세요. 
            전문 리크루터와 커리어 전문가를 위한 통합 플랫폼입니다.
          </p>
          
          <div className="branding-stats">
            <div className="glass-card">
              <span className="material-symbols-outlined glass-card-icon">rocket_launch</span>
              <p className="glass-card-num">95%</p>
              <p className="glass-card-label">채용 매칭 성공률</p>
            </div>
            <div className="glass-card">
              <span className="material-symbols-outlined glass-card-icon">speed</span>
              <p className="glass-card-num">2.4배</p>
              <p className="glass-card-label">프로세스 가속화</p>
            </div>
          </div>
        </div>
        
        <img 
          className="branding-bg-img"
          alt="Modern corporate office interior" 
          src="https://lh3.googleusercontent.com/aida-public/AB6AXuAdlCKlZjkkH4zQm3xGz-NbGmccdgB5LLz46Gd7penCAySZAg2n1HEA48xjNyanPc9-fwNNrqN0tXrfggMVuh8ugZ3RH1e-8VUa7obtsofYJIikZKBiotKIRzuU0dmYnR2HWAY3GKZkSUzdGxRw0usSOHO9RE3ZWoYq9HWd_Q4JWRT-CQFNi1Txu5Ya2PaRuPbn0G_HqfMacHCL4gUNlmZyr2oEn4A-2LJL16IxnrgLfPL6oZavTOAizJ5t-8v1XCQxYo2wNcRCXNAt"
        />
      </div>

      {/* Right Interactive Panel (Login Form) */}
      <div className="form-panel">
        <header className="form-header">
          <span className="material-symbols-outlined logo-icon">work</span>
          <span className="logo-text">RecruitMaster</span>
        </header>

        <main className="form-main">
          <div className="form-title-group">
            <h2 className="form-title">로그인</h2>
            <p className="form-subtitle">플랫폼의 모든 기능을 사용하려면 로그인하세요.</p>
          </div>

          {/* Feedback Messages */}
          {errorMessage && (
            <div style={{
              backgroundColor: 'var(--color-error-container)',
              color: 'var(--color-on-error-container)',
              padding: '12px',
              borderRadius: 'var(--radius-default)',
              fontSize: '13px',
              marginBottom: '16px',
              border: '1px solid var(--color-error)',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>error</span>
              {errorMessage}
            </div>
          )}

          {successMessage && (
            <div style={{
              backgroundColor: '#e6f4ea',
              color: '#137333',
              padding: '12px',
              borderRadius: 'var(--radius-default)',
              fontSize: '13px',
              marginBottom: '16px',
              border: '1px solid #137333',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>check_circle</span>
              {successMessage}
            </div>
          )}

          <form className="login-form" onSubmit={handleLogin}>
            {/* Email Input */}
            <div className="input-group">
              <label className="input-label" htmlFor="email">이메일 주소</label>
              <div className={`input-wrapper ${emailFocus ? 'focus-within' : ''}`}>
                <span className="material-symbols-outlined input-field-icon">mail</span>
                <input 
                  className="input-field" 
                  id="email" 
                  name="email" 
                  placeholder="example@recruit.com" 
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  onFocus={() => setEmailFocus(true)}
                  onBlur={() => setEmailFocus(false)}
                />
              </div>
            </div>

            {/* Password Input */}
            <div className="input-group">
              <div className="input-label-row">
                <label className="input-label" htmlFor="password">비밀번호</label>
                <a className="forgot-pwd-link" href="#forgot" onClick={(e) => { e.preventDefault(); alert('비밀번호 찾기 페이지로 이동합니다.'); }}>
                  비밀번호를 잊으셨나요?
                </a>
              </div>
              <div className={`input-wrapper ${pwdFocus ? 'focus-within' : ''}`}>
                <span className="material-symbols-outlined input-field-icon">lock</span>
                <input 
                  className="input-field" 
                  id="password" 
                  name="password" 
                  placeholder="••••••••" 
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onFocus={() => setPwdFocus(true)}
                  onBlur={() => setPwdFocus(false)}
                />
                <button 
                  className="pwd-toggle-btn" 
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? '비밀번호 숨기기' : '비밀번호 보기'}
                >
                  <span className="material-symbols-outlined">
                    {showPassword ? 'visibility_off' : 'visibility'}
                  </span>
                </button>
              </div>
            </div>

            {/* Remember Me */}
            <div className="remember-me-group" onClick={() => setRememberMe(!rememberMe)}>
              <input 
                className="checkbox-input" 
                id="remember" 
                name="remember" 
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                onClick={(e) => e.stopPropagation()}
              />
              <label className="checkbox-label" htmlFor="remember">이메일 기억하기</label>
            </div>

            {/* Action Buttons */}
            <div className="form-actions">
              <button 
                className="btn btn-primary" 
                type="submit"
                disabled={isSubmitting}
              >
                {isSubmitting ? '로그인 중...' : '확인'}
                {!isSubmitting && <span className="material-symbols-outlined">login</span>}
              </button>
              <button 
                className="btn btn-secondary" 
                type="button"
                onClick={handleCancel}
                disabled={isSubmitting}
              >
                취소
              </button>
            </div>
          </form>

          {/* Social Login Divider */}
          <div className="social-divider">
            <div className="divider-line"></div>
            <span className="divider-text">간편 로그인</span>
            <div className="divider-line"></div>
          </div>

          {/* Social Buttons */}
          <div className="social-buttons">
            <button className="btn-social" onClick={() => handleSocialLogin('Google')} aria-label="Google 로그인">
              <span className="material-symbols-outlined">account_circle</span>
            </button>
            <button className="btn-social" onClick={() => handleSocialLogin('Kakao')} aria-label="카카오 로그인">
              <span className="material-symbols-outlined">chat</span>
            </button>
          </div>

          {/* Registration Link */}
          <div className="register-group">
            <p className="register-text">
              계정이 없으신가요? 
              <a className="register-link" href="#signup" onClick={(e) => { e.preventDefault(); alert('회원가입 페이지로 이동합니다.'); }}>
                회원가입
              </a>
            </p>
          </div>
        </main>

        <footer className="form-footer">
          <p className="footer-text">© 2024 RecruitMaster. All rights reserved.</p>
        </footer>
      </div>
    </div>
  )
}

export default App
