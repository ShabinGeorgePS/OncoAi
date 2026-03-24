import streamlit as st

# ── Dummy credentials (replace with DB later) ─────────────────────────────────
DUMMY_USERS = {
    "admin@oncoai.com":   {"password": "admin123",   "role": "Admin",   "name": "Admin"},
    "doctor@oncoai.com":  {"password": "doctor123",  "role": "Doctor",  "name": "Doctor"},
    "patient@oncoai.com": {"password": "patient123", "role": "Patient", "name": "Patient"},
}

# ── Role colors and icons ──────────────────────────────────────────────────────
ROLE_CONFIG = {
    "Admin":   {"color": "#1a3a36", "bg": "#e8f4f2", "border": "#1a6b5e",
                "icon": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="#1a6b5e" viewBox="0 0 512 512"><path d="M495.9 166.6c3.2 8.7 .5 18.4-6.4 24.6l-43.3 39.4c1.1 8.3 1.7 16.8 1.7 25.4s-.6 17.1-1.7 25.4l43.3 39.4c6.9 6.2 9.6 15.9 6.4 24.6c-4.4 11.9-9.7 23.3-15.8 34.3l-4.7 8.1c-6.6 11-14 21.4-22.1 31.2c-5.9 7.2-15.7 9.6-24.5 6.8l-55.7-17.7c-13.4 10.3-28.2 18.9-44 25.4l-12.5 57.1c-2 9.1-9 16.3-18.2 17.8c-13.8 2.3-28 3.5-42.5 3.5s-28.7-1.2-42.5-3.5c-9.2-1.5-16.2-8.7-18.2-17.8l-12.5-57.1c-15.8-6.5-30.6-15.1-44-25.4L83.1 425.9c-8.8 2.8-18.6 .3-24.5-6.8c-8.1-9.8-15.5-20.2-22.1-31.2l-4.7-8.1c-6.1-11-11.4-22.4-15.8-34.3c-3.2-8.7-.5-18.4 6.4-24.6l43.3-39.4C64.6 273.1 64 264.6 64 256s.6-17.1 1.7-25.4L22.4 191.2c-6.9-6.2-9.6-15.9-6.4-24.6c4.4-11.9 9.7-23.3 15.8-34.3l4.7-8.1c6.6-11 14-21.4 22.1-31.2c5.9-7.2 15.7-9.6 24.5-6.8l55.7 17.7c13.4-10.3 28.2-18.9 44-25.4l12.5-57.1c2-9.1 9-16.3 18.2-17.8C227.3 1.2 241.5 0 256 0s28.7 1.2 42.5 3.5c9.2 1.5 16.2 8.7 18.2 17.8l12.5 57.1c15.8 6.5 30.6 15.1 44 25.4l55.7-17.7c8.8-2.8 18.6-.3 24.5 6.8c8.1 9.8 15.5 20.2 22.1 31.2l4.7 8.1c6.1 11 11.4 22.4 15.8 34.3zM256 336a80 80 0 1 0 0-160 80 80 0 1 0 0 160z"/></svg>'},
    "Doctor":  {"color": "#1a3a96", "bg": "#eff6ff", "border": "#2563eb",
                "icon": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="#2563eb" viewBox="0 0 576 512"><path d="M142.4 21.9c5.6 16.8-3.5 34.9-20.2 40.5L96 71.1 96 192c0 53 43 96 96 96s96-43 96-96l0-120.9-26.1-8.7c-16.8-5.6-25.8-23.7-20.2-40.5s23.7-25.8 40.5-20.2l26.1 8.7C334.4 19.1 352 43.5 352 71.1L352 192c0 77.2-54.6 141.6-127.3 156.7C231 404.6 278.4 448 336 448c61.9 0 112-50.1 112-112l0-70.7c-28.3-12.3-48-40.5-48-73.3c0-44.2 35.8-80 80-80s80 35.8 80 80c0 32.8-19.7 61-48 73.3l0 70.7c0 97.2-78.8 176-176 176c-92.9 0-168.9-71.9-175.5-163.1C87.2 334.2 32 269.6 32 192L32 71.1c0-27.5 17.6-52 43.9-60.4l26.1-8.7c16.8-5.6 34.9 3.5 40.5 20.2z"/></svg>'},
    "Patient": {"color": "#7d1a1a", "bg": "#fff5f5", "border": "#c0392b",
                "icon": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="#c0392b" viewBox="0 0 448 512"><path d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512l388.6 0c16.4 0 29.7-13.3 29.7-29.7C448 383.8 368.2 304 269.7 304l-91.4 0z"/></svg>'},
}

def init_session():
    """Initialize session state variables."""
    defaults = {
        "logged_in":    False,
        "user_role":    None,
        "user_name":    None,
        "user_email":   None,
        "auth_page":    "login",   # login | signup | forgot
        "auth_message": None,
        "auth_error":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def logout():
    """Clear session and return to login."""
    for key in ["logged_in", "user_role", "user_name", "user_email"]:
        st.session_state[key] = None if key != "logged_in" else False
    st.session_state.auth_page    = "login"
    st.session_state.auth_message = None
    st.session_state.auth_error   = None
    st.rerun()


def show_auth_page():
    """Render the full auth UI. Returns True if user is logged in."""
    init_session()

    if st.session_state.logged_in:
        return True

    # ── Page config already set in app.py, just inject auth CSS ───────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    [data-testid="stAppViewContainer"] { background: #f5f7f6; font-family:'DM Sans',sans-serif; }
    [data-testid="stSidebar"]          { display: none !important; }

    /* Center the auth card */
    .auth-outer {
        display: flex; align-items: center; justify-content: center;
        min-height: 88vh; padding: 2rem 0;
    }
    .auth-card {
        background: #ffffff;
        border: 1px solid #e2e8e6;
        border-radius: 18px;
        padding: 2.4rem 2.8rem;
        width: 100%; max-width: 460px;
        box-shadow: 0 4px 24px rgba(26,107,94,0.10);
    }
    .auth-logo {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem; font-weight: 900;
        color: #1a6b5e; margin: 0 0 0.1rem 0;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .auth-tagline {
        font-size: 0.8rem; color: #7a9e97; font-weight: 500;
        letter-spacing: 0.06em; text-transform: uppercase;
        margin: 0 0 1.8rem 0;
    }
    .auth-title {
        font-size: 1.4rem; font-weight: 700; color: #1a3a36;
        margin: 0 0 0.3rem 0; font-family: 'DM Serif Display', serif;
    }
    .auth-sub {
        font-size: 0.85rem; color: #7a9e97; margin: 0 0 1.5rem 0;
    }
    .role-grid {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 0.6rem; margin-bottom: 1.2rem;
    }
    .role-card {
        border: 2px solid #e2e8e6; border-radius: 10px;
        padding: 0.7rem 0.5rem; text-align: center; cursor: pointer;
        transition: all 0.15s;
    }
    .role-card.selected {
        border-color: #1a6b5e; background: #e8f4f2;
    }
    .role-card:hover { border-color: #b8dbd6; }
    .role-label {
        font-size: 0.78rem; font-weight: 600; color: #4a5568;
        margin: 0.3rem 0 0 0;
    }
    .auth-divider {
        border: none; border-top: 1px solid #e8f0ee;
        margin: 1.2rem 0;
    }
    .auth-link {
        font-size: 0.82rem; color: #1a6b5e; cursor: pointer;
        font-weight: 600; text-decoration: underline;
    }
    .auth-link:hover { color: #028090; }
    .success-banner {
        background: #e8f8f5; border: 1px solid #b8dbd6;
        border-left: 4px solid #1a6b5e; border-radius: 8px;
        padding: 0.7rem 1rem; font-size: 0.85rem; color: #1a3a36;
        margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;
    }
    .error-banner {
        background: #fff5f5; border: 1px solid #f5c6c6;
        border-left: 4px solid #c0392b; border-radius: 8px;
        padding: 0.7rem 1rem; font-size: 0.85rem; color: #7b241c;
        margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;
    }
    .demo-box {
        background: #fafcfb; border: 1px dashed #b8dbd6;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.78rem; color: #4a7a73; margin-top: 1rem;
        line-height: 1.9;
    }
    .demo-box b { color: #1a3a36; }
    /* Override streamlit input styles */
    [data-testid="stTextInput"] input {
        border: 1px solid #e2e8e6 !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 0.8rem !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #1a6b5e !important;
        box-shadow: 0 0 0 2px rgba(26,107,94,0.15) !important;
    }
    [data-testid="stSelectbox"] > div > div {
        border: 1px solid #e2e8e6 !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: #1a6b5e !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 1.5rem !important;
        width: 100% !important;
        transition: background 0.2s !important;
    }
    .stButton > button:hover { background: #145a4e !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Logo header ────────────────────────────────────────────────────────────
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div style="text-align:center; padding: 2rem 0 0 0;">
            <p class="auth-logo" style="justify-content:center;">
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28"
                     fill="#1a6b5e" viewBox="0 0 512 512">
                    <path d="M160 96a96 96 0 1 1 192 0A96 96 0 1 1 160 96z"/>
                </svg>
                ONCOAi
            </p>
            <p class="auth-tagline">AI-Powered Oral Cancer Detection</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Banners ────────────────────────────────────────────────────────────
        if st.session_state.auth_message:
            st.markdown(f"""
            <div class="success-banner">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                     fill="#1a6b5e" viewBox="0 0 512 512">
                    <path d="M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zM369
                             209L241 337c-9.4 9.4-24.6 9.4-33.9 0l-64-64c-9.4-9.4-9.4-24.6
                             0-33.9s24.6-9.4 33.9 0l47 47L335 175c9.4-9.4 24.6-9.4 33.9
                             0s9.4 24.6 0 33.9z"/>
                </svg>
                {st.session_state.auth_message}
            </div>
            """, unsafe_allow_html=True)
            st.session_state.auth_message = None

        if st.session_state.auth_error:
            st.markdown(f"""
            <div class="error-banner">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                     fill="#c0392b" viewBox="0 0 512 512">
                    <path d="M256 32c14.2 0 27.3 7.5 34.5 19.8l216 368c7.3 12.4 7.3
                             27.7.2 40.1S486.3 480 472 480L40 480c-14.3 0-27.6-7.7-34.7-20.1
                             s-7-27.8.2-40.1l216-368C228.7 39.5 241.8 32 256 32zm0 128
                             c-13.3 0-24 10.7-24 24l0 112c0 13.3 10.7 24 24 24s24-10.7
                             24-24l0-112c0-13.3-10.7-24-24-24zm32 224a32 32 0 1 0-64 0
                             32 32 0 1 0 64 0z"/>
                </svg>
                {st.session_state.auth_error}
            </div>
            """, unsafe_allow_html=True)
            st.session_state.auth_error = None

        # ── AUTH PAGES ─────────────────────────────────────────────────────────
        page = st.session_state.auth_page

        # ══════════════════════════════════════════════════════════════════════
        if page == "login":
        # ══════════════════════════════════════════════════════════════════════
            st.markdown('<p class="auth-title">Welcome back</p>', unsafe_allow_html=True)
            st.markdown('<p class="auth-sub">Sign in to your ONCOAi account</p>', unsafe_allow_html=True)

            email    = st.text_input("Email Address", placeholder="you@example.com", key="login_email")
            password = st.text_input("Password",      placeholder="••••••••",        type="password", key="login_pw")

            col1, col2 = st.columns([1, 1])
            with col2:
                if st.button("Forgot Password?", key="to_forgot",
                             help="Reset your password"):
                    st.session_state.auth_page = "forgot"
                    st.rerun()

            if st.button("Sign In", key="login_btn"):
                if not email or not password:
                    st.session_state.auth_error = "Please enter both email and password."
                    st.rerun()
                elif email in DUMMY_USERS and DUMMY_USERS[email]["password"] == password:
                    user = DUMMY_USERS[email]
                    st.session_state.logged_in  = True
                    st.session_state.user_role  = user["role"]
                    st.session_state.user_name  = user["name"]
                    st.session_state.user_email = email
                    st.session_state.auth_message = None
                    st.rerun()
                else:
                    st.session_state.auth_error = "Invalid email or password. Please try again."
                    st.rerun()

            st.markdown('<hr class="auth-divider">', unsafe_allow_html=True)
            st.markdown(
                '<p style="text-align:center; font-size:0.85rem; color:#7a9e97;">'
                'Don\'t have an account? </p>',
                unsafe_allow_html=True
            )
            if st.button("Create Account", key="to_signup"):
                st.session_state.auth_page = "signup"
                st.rerun()

            # Demo credentials
            st.markdown("""
            <div class="demo-box">
                <b>Demo Credentials:</b><br>
                🔧 Admin &nbsp;&nbsp;→ admin@oncoai.com &nbsp;/ admin123<br>
                🩺 Doctor &nbsp;→ doctor@oncoai.com / doctor123<br>
                🧑 Patient → patient@oncoai.com / patient123
            </div>
            """, unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        elif page == "signup":
        # ══════════════════════════════════════════════════════════════════════
            st.markdown('<p class="auth-title">Create Account</p>', unsafe_allow_html=True)
            st.markdown('<p class="auth-sub">Join ONCOAi — select your role to get started</p>',
                        unsafe_allow_html=True)

            # Role selector
            role = st.selectbox(
                "Select Role",
                options=["Patient", "Doctor", "Admin"],
                key="signup_role"
            )
            cfg = ROLE_CONFIG[role]

            # Role info badge
            st.markdown(f"""
            <div style="background:{cfg['bg']}; border:1px solid {cfg['border']};
            border-radius:8px; padding:0.5rem 0.8rem; font-size:0.8rem;
            color:{cfg['color']}; margin-bottom:0.8rem; display:flex;
            align-items:center; gap:0.5rem;">
                {cfg['icon']}
                <span><b>{role} Account:</b>
                {'Full system access and user management.' if role=='Admin'
                 else 'Access to patient records and diagnostic tools.' if role=='Doctor'
                 else 'Upload images and view your detection results.'}
                </span>
            </div>
            """, unsafe_allow_html=True)

            name     = st.text_input("Full Name",        placeholder="Your full name",     key="signup_name")
            email    = st.text_input("Email Address",    placeholder="you@example.com",    key="signup_email")
            phone    = st.text_input("Phone Number",     placeholder="+91 XXXXX XXXXX",    key="signup_phone")
            password = st.text_input("Password",         placeholder="Min 8 characters",   type="password", key="signup_pw")
            confirm  = st.text_input("Confirm Password", placeholder="Repeat password",    type="password", key="signup_cpw")

            if role == "Doctor":
                st.text_input("Medical Registration Number", placeholder="MCI-XXXX-XXXX", key="signup_mci")

            if st.button("Create Account", key="signup_btn"):
                if not all([name, email, password, confirm]):
                    st.session_state.auth_error = "Please fill in all required fields."
                elif password != confirm:
                    st.session_state.auth_error = "Passwords do not match."
                elif len(password) < 8:
                    st.session_state.auth_error = "Password must be at least 8 characters."
                elif "@" not in email:
                    st.session_state.auth_error = "Please enter a valid email address."
                else:
                    # In dummy mode — just show success and redirect to login
                    st.session_state.auth_message = (
                        f"Account created for {name}! "
                        "Please sign in. (Backend not connected yet)"
                    )
                    st.session_state.auth_page = "login"
                st.rerun()

            st.markdown('<hr class="auth-divider">', unsafe_allow_html=True)
            st.markdown(
                '<p style="text-align:center; font-size:0.85rem; color:#7a9e97;">'
                'Already have an account? </p>',
                unsafe_allow_html=True
            )
            if st.button("Sign In Instead", key="to_login_from_signup"):
                st.session_state.auth_page = "login"
                st.rerun()

        # ══════════════════════════════════════════════════════════════════════
        elif page == "forgot":
        # ══════════════════════════════════════════════════════════════════════
            st.markdown('<p class="auth-title">Reset Password</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="auth-sub">Enter your email and we\'ll send a reset link</p>',
                unsafe_allow_html=True
            )

            email = st.text_input("Email Address", placeholder="you@example.com", key="forgot_email")

            if st.button("Send Reset Link", key="forgot_btn"):
                if not email or "@" not in email:
                    st.session_state.auth_error = "Please enter a valid email address."
                else:
                    st.session_state.auth_message = (
                        f"Reset link sent to {email}. "
                        "Check your inbox. (Backend not connected yet)"
                    )
                    st.session_state.auth_page = "login"
                st.rerun()

            st.markdown('<hr class="auth-divider">', unsafe_allow_html=True)
            if st.button("← Back to Sign In", key="back_to_login"):
                st.session_state.auth_page = "login"
                st.rerun()

        # ── Footer ─────────────────────────────────────────────────────────────
        st.markdown("""
        <p style="text-align:center; font-size:0.72rem; color:#b8dbd6;
        margin-top:1.5rem;">
            © 2026 ONCOAi · Team MediScope · SKCT Coimbatore
        </p>
        """, unsafe_allow_html=True)

    return False  # Not logged in yet


def show_user_badge():
    """Show logged-in user info + logout in sidebar."""
    if not st.session_state.get("logged_in"):
        return

    role = st.session_state.user_role
    cfg  = ROLE_CONFIG.get(role, ROLE_CONFIG["Patient"])

    with st.sidebar:
        st.markdown(f"""
        <div style="background:{cfg['bg']}; border:1px solid {cfg['border']};
        border-radius:10px; padding:0.8rem 1rem; margin-bottom:1rem;">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">
                {cfg['icon']}
                <span style="font-weight:700; color:{cfg['color']};
                font-size:0.85rem;">{role}</span>
            </div>
            <p style="font-size:0.88rem; font-weight:600; color:#1a3a36; margin:0;">
                {st.session_state.user_name}
            </p>
            <p style="font-size:0.75rem; color:#7a9e97; margin:0;">
                {st.session_state.user_email}
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Sign Out", key="logout_btn", use_container_width=True):
            logout()
