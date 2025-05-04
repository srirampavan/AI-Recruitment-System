nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Initialize session state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_type" not in st.session_state:
    st.session_state["user_type"] = None  # 'candidate', 'recruiter', 'admin'

# Constants
DATA_FILE = "data.json"
JOB_DESCRIPTION_DIR = "Data/JobDesc/"
UPLOADS_DIR = "uploads/"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize database
def initialize_database():
    if not os.path.exists(DATA_FILE):
        data = {
            "users": [],
            "recruiters": [],
            "admins": [{
                "username": "admin",
                "password": hashlib.sha256("admin123".encode()).hexdigest(),
                "email": "admin@recruit.com"
            }],
            "jobs": [],
            "applications": []
        }
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=4)

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# User management
def create_account(name, email, password, user_type="candidate", additional_info=None):
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        
        # Check if email already exists
        if any(user["email"] == email for user in data["users"] + data["recruiters"] + data["admins"]):
            return False, "Email already exists"
        
        user_data = {
            "name": name,
            "email": email,
            "password": hash_password(password),
            "created_at": datetime.now().isoformat(),
            "user_type": user_type,
            "resume": None,
            "resume_path": None,
            "profile_complete": False
        }
        
        if user_type == "candidate" and additional_info:
            user_data.update({
                "age": additional_info.get("age"),
                "sex": additional_info.get("sex"),
                "education": additional_info.get("education"),
                "experience": additional_info.get("experience")
            })
        
        if user_type == "candidate":
            data["users"].append(user_data)
        elif user_type == "recruiter":
            data["recruiters"].append(user_data)
        
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=4)
        
        return True, "Account created successfully"
    except Exception as e:
        return False, f"Error creating account: {str(e)}"

def authenticate_user(email, password, user_type="candidate"):
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        
        user_list = data["users"] if user_type == "candidate" else (
            data["recruiters"] if user_type == "recruiter" else data["admins"]
        )
        
        for user in user_list:
            if user["email"] == email and user["password"] == hash_password(password):
                return True, user
        
        return False, "Invalid credentials"
    except Exception as e:
        return False, f"Authentication error: {str(e)}"

# File processing
def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def extract_text(file_path):
    try:
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                return '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
        elif extension == '.docx':
            doc = docx.Document(file_path)
            return '\n'.join(para.text for para in doc.paragraphs)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

# NLP Processing
def preprocess_text(text):
    try:
        # Tokenization
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation and non-alphabetic tokens
        tokens = [word for word in tokens if word.isalpha()]
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        return tokens
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return []

def extract_keywords(text, n=10):
    tokens = preprocess_text(text)
    freq_dist = FreqDist(tokens)
    return [word for word, _ in freq_dist.most_common(n)]

def calculate_similarity(job_desc_text, resume_text):
    job_tokens = set(preprocess_text(job_desc_text))
    resume_tokens = set(preprocess_text(resume_text))
    
    if not job_tokens:
        return 0
    
    intersection = job_tokens.intersection(resume_tokens)
    return round((len(intersection) / len(job_tokens)) * 100, 2)

# AI Integration
def analyze_resume_with_ai(resume_text, job_desc_text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    prompt = f"""
    Analyze this resume against the job description and provide:
    1. A match score (0-100)
    2. Top 5 strengths
    3. Top 3 weaknesses
    4. TOP 2 Suggestions for improvement
    
    Resume:
    {resume_text[:3000]}  # Limiting to first 3000 chars to avoid token limits
    
    Job Description:
    {job_desc_text[:2000]}
    
    Provide output in JSON format with keys: score, strengths, weaknesses, suggestions.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        analysis= json.loads(response.choices[0].message.content)
        # Add visualization
        st.subheader("Match Breakdown")
        match_score = analysis['score']
        non_match = 100 - match_score
        fig = px.pie(
            values=[match_score, non_match],
            names=['Match', 'Non-Match'],
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return analysis
    
    except Exception as e:
        st.error(f"AI analysis error: {str(e)}")
        return None

def generate_interview_question(resume_text, job_desc_text="General technical screening questions"):
    """Generate one interview question based on resume"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    prompt = f"""
    Generate one technical interview question based on this resume:
    {resume_text[:2000]}
    
    Job Context: {job_desc_text}
    
    Return ONLY the question text.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating question: {str(e)}")
        return "Could not generate question at this time."

def generate_score(question, answer, resume):
    """Generate score (1-10) for an answer"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    prompt = f"""
    Rate this interview answer (1-10) based on:
    - Technical accuracy (40%)
    - Relevance to question (30%) 
    - Clarity (30%)
    
    Question: {question}
    Answer: {answer}
    Resume Context: {resume[:1000]}
    
    Return ONLY the score as a number between 1-10.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temp for consistent scoring
            max_tokens=2
        )
        score = float(response.choices[0].message.content.strip())
        return max(1, min(10, score))  # Ensure score is between 1-10
    except Exception:
        return 5.0  # Default score if error occurs
def login_page():
    st.title("AI Recruitment System - Login")
    
    user_type = st.radio("Login as:", ("Candidate", "Recruiter", "Admin"))
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            success, result = authenticate_user(
                email, 
                password, 
                user_type.lower()
            )
            
            if success:
                st.session_state["logged_in"] = True
                st.session_state["user_info"] = result
                st.session_state["user_type"] = user_type.lower()
                st.success("Login successful!")
                st.rerun()
            else:
                st.error(result)

def candidate_signup_page():
    st.title("Candidate Registration")
    
    with st.form("candidate_signup"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        # Additional candidate info
        age = st.number_input("Age", min_value=18, max_value=100)
        sex = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        education = st.text_input("Highest Education")
        experience = st.text_input("Years of Experience")
        
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
                return
                
            additional_info = {
                "age": age,
                "sex": sex,
                "education": education,
                "experience": experience
            }
            
            success, message = create_account(
                name, email, password, 
                user_type="candidate",
                additional_info=additional_info
            )
            
            if success:
                st.success(message)
                st.session_state["show_login"] = True
            else:
                st.error(message)

def recruiter_signup_page():
    st.title("Recruiter Registration")
    
    with st.form("recruiter_signup"):
        name = st.text_input("Full Name")
        email = st.text_input("Company Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        company = st.text_input("Company Name")
        
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
                return
                
            success, message = create_account(
                name, email, password, 
                user_type="recruiter",
                additional_info={"company": company}
            )
            
            if success:
                st.success(message)
                st.session_state["show_login"] = True
            else:
                st.error(message)

def candidate_dashboard():
    user_info = st.session_state["user_info"]
    st.title(f"Welcome, {user_info['name']}")
    
    tabs = st.tabs(["Profile", "Resume Analysis", "Interview Prep", "Applications"])
    
    with tabs[0]:  # Profile tab
        st.subheader("Your Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {user_info['name']}")
            st.write(f"**Email:** {user_info['email']}")
            st.write(f"**Age:** {user_info.get('age', 'Not specified')}")
            st.write(f"**Gender:** {user_info.get('sex', 'Not specified')}")
        
        with col2:
            st.write(f"**Education:** {user_info.get('education', 'Not specified')}")
            st.write(f"**Experience:** {user_info.get('experience', 'Not specified')}")
            st.write(f"**Profile Complete:** {'Yes' if user_info.get('profile_complete', False) else 'No'}")
        
        if st.button("Edit Profile"):
            st.session_state["edit_profile"] = True
    
    with tabs[1]:  # Resume Analysis tab
        st.subheader("Resume Analysis")
        
        uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
        
        if uploaded_file:
            file_path = save_uploaded_file(uploaded_file)
            resume_text = extract_text(file_path)
            
            if resume_text:
                # Save to user profile
                with open(DATA_FILE, "r+") as f:
                    data = json.load(f)
                    for user in data["users"]:
                        if user["email"] == user_info["email"]:
                            user["resume"] = resume_text
                            user["resume_path"] = file_path
                            user["profile_complete"] = True
                            break
                    
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
                
                st.session_state["user_info"]["resume"] = resume_text
                st.session_state["user_info"]["resume_path"] = file_path
                st.session_state["user_info"]["profile_complete"] = True
                st.success("Resume uploaded successfully!")
                
                # Show resume keywords
                st.subheader("Extracted Keywords")
                keywords = extract_keywords(resume_text)
                st.write(", ".join(keywords))
                
                # Job selection for analysis
                with open(DATA_FILE, "r") as f:
                    data = json.load(f)
                    jobs = data["jobs"]
                
                if jobs:
                    job_titles = [job["title"] for job in jobs]
                    selected_job = st.selectbox("Select a job to analyze against", job_titles)
                    
                    if selected_job:
                        job_desc = next(job for job in jobs if job["title"] == selected_job)
                        st.subheader(f"Analysis for: {selected_job}")
                        
                        # Calculate basic similarity score
                        similarity_score = calculate_similarity(job_desc["description"], resume_text)
                        st.metric("Basic Match Score", f"{similarity_score}%")
                        
                        # AI analysis
                        if st.button("Run Advanced AI Analysis"):
                            with st.spinner("Analyzing resume with AI..."):
                                analysis = analyze_resume_with_ai(resume_text, job_desc["description"])
                                
                                if analysis:
                                    st.subheader("AI Analysis Results")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.metric("AI Match Score", f"{analysis['score']}%")
                                        
                                        st.subheader("Strengths")
                                        for strength in analysis["strengths"]:
                                            st.success(f"✓ {strength}")
                                    
                                    with col2:
                                        st.subheader("Weaknesses")
                                        for weakness in analysis["weaknesses"]:
                                            st.error(f"✗ {weakness}")
                                        
                                        st.subheader("Suggestions")
                                        for suggestion in analysis["suggestions"]:
                                            st.info(f"• {suggestion}")
                                    
                                    # Save application
                                    application = {
                                        "candidate_email": user_info["email"],
                                        "job_id": job_desc["id"],
                                        "basic_score": similarity_score,
                                        "ai_score": analysis["score"],
                                        "applied_at": datetime.now().isoformat(),
                                        "status": "analyzed"
                                    }
                                    
                                    with open(DATA_FILE, "r+") as f:
                                        data = json.load(f)
                                        data["applications"].append(application)
                                        f.seek(0)
                                        json.dump(data, f, indent=4)
                                        f.truncate()
                else:
                    st.warning("No jobs available for analysis")
    
    with tabs[2]:  # Interview Prep tab
        st.subheader("Interview Preparation")
        
        if not user_info.get("resume"):
            st.warning("Please upload your resume first")
            return
        
        # Initialize session state
        if "interview_data" not in st.session_state:
            st.session_state.interview_data = {
                "in_progress": False,
                "questions": [],
                "current_q": 0,
                "scores": []
            }
        
        # Start interview
        if not st.session_state.interview_data["in_progress"] and st.button("Start Interview"):
            with st.spinner("Preparing your interview..."):
                st.session_state.interview_data = {
                    "in_progress": True,
                    "questions": [generate_interview_question(user_info["resume"]) for _ in range(3)],
                    "current_q": 0,
                    "scores": []
                }
            st.rerun()
        
        # Interview in progress
        if st.session_state.interview_data["in_progress"]:
            data = st.session_state.interview_data
            current_q = data["current_q"]
            
            # Show current question
            st.progress((current_q + 1) / len(data["questions"]))
            st.write(f"**Question {current_q + 1}/{len(data['questions'])}**")
            st.write(data["questions"][current_q])
            
            # Answer input
            answer = st.text_area("Your answer", key=f"answer_{current_q}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Answer"):
                    if answer.strip():
                        # Generate and store score
                        score = generate_score(
                            data["questions"][current_q],
                            answer,
                            user_info["resume"]
                        )
                        data["scores"].append(score)
                        data["current_q"] += 1
                        
                        if data["current_q"] >= len(data["questions"]):
                            # Interview complete - show results
                            st.session_state.interview_data["in_progress"] = False
                            st.success("Interview Completed!")
                            
                            # Display scores
                            st.subheader("Your Scores")
                            for i, (q, s) in enumerate(zip(data["questions"], data["scores"])):
                                st.write(f"Q{i+1}: {s}/10 - {q[:50]}...")
                            
                            avg_score = sum(data["scores"]) / len(data["scores"])
                            st.metric("Average Score", f"{avg_score:.1f}/10")
                        else:
                            st.rerun()
                    else:
                        st.warning("Please enter your answer")
            
            with col2:
                if st.button("Cancel Interview"):
                    st.session_state.interview_data = {
                        "in_progress": False,
                        "questions": [],
                        "current_q": 0,
                        "scores": []
                    }
                    st.rerun()
        else:
            st.info("Click 'Start Interview' to begin your practice session")

        
    with tabs[3]:  # Applications tab
        st.subheader("Your Applications")
        
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            user_applications = [
                app for app in data["applications"] 
                if app["candidate_email"] == user_info["email"]
            ]
            
            if user_applications:
                for app in user_applications:
                    job = next(j for j in data["jobs"] if j["id"] == app["job_id"])
                    
                    with st.expander(f"{job['title']} - {app['status']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Basic Score", f"{app['basic_score']}%")
                        
                        with col2:
                            st.metric("AI Score", f"{app.get('ai_score', 'N/A')}%")
                        
                        st.write(f"**Applied on:** {app['applied_at'][:10]}")
                        st.write(f"**Status:** {app['status'].capitalize()}")
            else:
                st.info("You haven't applied to any jobs yet")

def recruiter_dashboard():
    user_info = st.session_state["user_info"]
    st.title(f"Recruiter Dashboard - {user_info.get('name', 'Recruiter')}")
    
    tabs = st.tabs(["Post Jobs", "View Applications", "Candidate Search"])
    
    with tabs[0]:  # Post Jobs tab
        st.subheader("Post a New Job")
        
        with st.form("job_post_form"):
            title = st.text_input("Job Title")
            description = st.text_area("Job Description", height=200)
            requirements = st.text_area("Requirements", height=150)
            location = st.text_input("Location")
            job_type = st.selectbox("Job Type", ["Full-time", "Part-time", "Contract", "Internship"])
            deadline = st.date_input("Application Deadline")
            
            submit = st.form_submit_button("Post Job")
            
            if submit:
                new_job = {
                    "id": str(len(st.session_state.get("jobs", [])) + 1),
                    "title": title,
                    "description": description,
                    "requirements": requirements,
                    "location": location,
                    "type": job_type,
                    "deadline": deadline.isoformat(),
                    "posted_by": user_info["email"],
                    "posted_at": datetime.now().isoformat(),
                    "status": "active"
                }
                
                with open(DATA_FILE, "r+") as f:
                    data = json.load(f)
                    data["jobs"].append(new_job)
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
                
                st.success("Job posted successfully!")
    
    with tabs[1]:  # View Applications tab
        st.subheader("Job Applications")
        
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            recruiter_jobs = [job for job in data["jobs"] if job["posted_by"] == user_info["email"]]
            
            if recruiter_jobs:
                selected_job = st.selectbox("Select a job", [job["title"] for job in recruiter_jobs])
                
                if selected_job:
                    job_id = next(job["id"] for job in recruiter_jobs if job["title"] == selected_job)
                    applications = [app for app in data["applications"] if app["job_id"] == job_id]
                    
                    if applications:
                        valid_applications = []
                        invalid_apps = 0
                        
                        # First pass to validate applications
                        for app in applications:
                            try:
                                candidate = next(u for u in data["users"] if u["email"] == app["candidate_email"])
                                valid_applications.append((app, candidate))
                            except StopIteration:
                                invalid_apps += 1
                        
                        if invalid_apps > 0:
                            st.warning(f"Skipping {invalid_apps} applications with missing candidate data")
                        
                        st.write(f"Showing {len(valid_applications)} valid applications")
                        
                        for i, (app, candidate) in enumerate(valid_applications):
                            with st.expander(f"{candidate['name']} - Score: {app['basic_score']}%"):
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.write(f"**Name:** {candidate['name']}")
                                    st.write(f"**Email:** {candidate['email']}")
                                    st.write(f"**Score:** {app['basic_score']}%")
                                    st.write(f"**Status:** {app['status']}")
                                
                                with col2:
                                    if st.button(
                                        "View Resume", 
                                        key=f"view_{app['candidate_email']}_{job_id}_{i}"
                                    ):
                                        if candidate.get("resume_path"):
                                            with open(candidate["resume_path"], "rb") as f:
                                                st.download_button(
                                                    label="Download Resume",
                                                    data=f,
                                                    file_name=os.path.basename(candidate["resume_path"]),
                                                    mime="application/octet-stream",
                                                    key=f"download_{app['candidate_email']}_{job_id}_{i}"
                                                )
                                        else:
                                            st.warning("No resume available")
                                    
                                    new_status = st.selectbox(
                                        "Update Status",
                                        ["analyzed", "shortlisted", "rejected", "hired"],
                                        index=["analyzed", "shortlisted", "rejected", "hired"].index(app["status"]),
                                        key=f"status_{app['candidate_email']}_{job_id}_{i}"
                                    )
                                    
                                    if st.button(
                                        "Update", 
                                        key=f"update_{app['candidate_email']}_{job_id}_{i}"
                                    ):
                                        with open(DATA_FILE, "r+") as f:
                                            data = json.load(f)
                                            for a in data["applications"]:
                                                if a["candidate_email"] == app["candidate_email"] and a["job_id"] == job_id:
                                                    a["status"] = new_status
                                                    break
                                            
                                            f.seek(0)
                                            json.dump(data, f, indent=4)
                                            f.truncate()
                                        
                                        st.success("Status updated!")
                                        st.rerun()
                    else:
                        st.info("No applications for this job yet")
            else:
                st.info("You haven't posted any jobs yet")
    
    with tabs[2]:  # Candidate Search tab
        st.subheader("Candidate Search")
        
        search_query = st.text_input("Search by skills or keywords")
        min_score = st.slider("Minimum Match Score", 0, 100, 50)
        
        if st.button("Search Candidates"):
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                
                candidates = []
                for user in data["users"]:
                    if user.get("resume"):
                        # Simple keyword matching (would be enhanced in production)
                        resume_text = user["resume"].lower()
                        query_match = any(word.lower() in resume_text for word in search_query.split()) if search_query else True
                        
                        if query_match:
                            candidates.append(user)
                
                if candidates:
                    st.write(f"Found {len(candidates)} matching candidates")
                    
                    for candidate in candidates:
                        with st.expander(candidate["name"]):
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                st.write(f"**Email:** {candidate['email']}")
                                st.write(f"**Experience:** {candidate.get('experience', 'N/A')}")
                                st.write(f"**Education:** {candidate.get('education', 'N/A')}")
                            
                            with col2:
                                if candidate.get("resume_path"):
                                    with open(candidate["resume_path"], "rb") as f:
                                        st.download_button(
                                            label="Download Resume",
                                            data=f,
                                            file_name=os.path.basename(candidate["resume_path"]),
                                            mime="application/octet-stream"
                                        )
                                else:
                                    st.warning("No resume available")
                else:
                    st.info("No candidates found matching your criteria")

def admin_dashboard():
    st.title("Admin Dashboard")
    
    tabs = st.tabs(["System Overview", "User Management", "Job Management"])
    
    with tabs[0]:  # System Overview
        st.subheader("System Statistics")
        
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Candidates", len(data["users"]))
            
            with col2:
                st.metric("Total Recruiters", len(data["recruiters"]))
            
            with col3:
                st.metric("Total Jobs Posted", len(data["jobs"]))
        
        st.subheader("Recent Activity")
        
        # Show recent applications
        applications = sorted(
            data["applications"],
            key=lambda x: x["applied_at"],
            reverse=True
        )[:10]
        
        for app in applications:
            candidate = next((u for u in data["users"] if u["email"] == app["candidate_email"]), None)
            job = next((j for j in data["jobs"] if j["id"] == app["job_id"]), None)
            
            if candidate and job:
                st.write(f"{candidate['name']} applied for {job['title']} (Score: {app['basic_score']}%)")
    
    with tabs[1]:  # User Management
        st.subheader("User Management")
        
        user_type = st.radio("User Type", ["Candidates", "Recruiters"])
        
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            
            users = data["users"] if user_type == "Candidates" else data["recruiters"]
            
            for user in users:
                with st.expander(user["name"]):
                    st.write(f"**Email:** {user['email']}")
                    st.write(f"**Registered:** {user['created_at'][:10]}")
                    
                    if st.button("Delete", key=f"delete_{user['email']}"):
                        with open(DATA_FILE, "r+") as f:
                            data = json.load(f)
                            
                            if user_type == "Candidates":
                                data["users"] = [u for u in data["users"] if u["email"] != user["email"]]
                            else:
                                data["recruiters"] = [u for u in data["recruiters"] if u["email"] != user["email"]]
                            
                            f.seek(0)
                            json.dump(data, f, indent=4)
                            f.truncate()
                        
                        st.success("User deleted")
                        st.rerun()
    
    with tabs[2]:  # Job Management
        st.subheader("Job Management")
        
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
            
            for job in data["jobs"]:
                recruiter = next(
                    (r for r in data["recruiters"] + data["admins"] if r["email"] == job["posted_by"]),
                    None
                )
                
                with st.expander(job["title"]):
                    st.write(f"**Posted by:** {recruiter['name'] if recruiter else 'Unknown'}")
                    st.write(f"**Posted on:** {job['posted_at'][:10]}")
                    st.write(f"**Status:** {job['status']}")
                    
                    if st.button("Deactivate", key=f"deactivate_{job['id']}"):
                        with open(DATA_FILE, "r+") as f:
                            data = json.load(f)
                            
                            for j in data["jobs"]:
                                if j["id"] == job["id"]:
                                    j["status"] = "inactive"
                                    break
                            
                            f.seek(0)
                            json.dump(data, f, indent=4)
                            f.truncate()
                        
                        st.success("Job deactivated")
                        st.rerun()

# Main App
def main():
    initialize_database()
    
    st.sidebar.title("AI Recruitment System")
    
    if st.session_state.get("logged_in"):
        user_type = st.session_state["user_type"]
        
        if user_type == "candidate":
            candidate_dashboard()
        elif user_type == "recruiter":
            recruiter_dashboard()
        elif user_type == "admin":
            admin_dashboard()
        
        st.sidebar.button("Logout", on_click=lambda: st.session_state.clear())
    else:
        page = st.sidebar.radio(
            "Go to",
            ["Login", "Candidate Signup", "Recruiter Signup"]
        )
        
        if page == "Login":
            login_page()
        elif page == "Candidate Signup":
            candidate_signup_page()
        elif page == "Recruiter Signup":
            recruiter_signup_page()

if __name__ == "__main__":
    main()