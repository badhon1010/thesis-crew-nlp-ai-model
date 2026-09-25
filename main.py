from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import model
import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("VITE_GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

app = FastAPI(title="Thesis Crew Custom AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class StudentProfile(BaseModel):
    skills: Optional[List[str]] = []
    researchInterests: Optional[str] = ""

class ResearchTopic(BaseModel):
    id: Optional[str] = ""
    title: Optional[str] = ""
    category: Optional[str] = ""
    requiredSkills: Optional[List[str]] = []
    description: Optional[str] = ""

class MatchRequest(BaseModel):
    studentProfile: StudentProfile
    researchTopic: ResearchTopic

class RecommendRequest(BaseModel):
    studentProfile: StudentProfile
    topics: List[ResearchTopic]

@app.get("/")
def read_root():
    return {"message": "Welcome to Thesis Crew Custom AI API!"}

@app.post("/api/analyze-match")
async def analyze_match(request: MatchRequest):
    student_skills = request.studentProfile.skills or []
    topic_skills = request.researchTopic.requiredSkills or []
    
    score = model.calculate_match_score(
        student_skills,
        request.studentProfile.researchInterests,
        request.researchTopic.title,
        topic_skills,
        request.researchTopic.description
    )
    
    strengths = []
    gaps = []
    
    gemini_error = ""
    if API_KEY:
        try:
            gemini_model = genai.GenerativeModel("gemini-3.6-flash", generation_config={"response_mime_type": "application/json"})
            prompt = f"""
            You are an expert academic advisor. 
            Analyze the fit between a student's profile and a research topic.
            
            Student Profile:
            - Skills: {', '.join(request.studentProfile.skills) if request.studentProfile.skills else 'None specified'}
            - Research Interests: {request.studentProfile.researchInterests or 'None specified'}
            
            Research Topic:
            - Title: {request.researchTopic.title or 'None specified'}
            - Category: {request.researchTopic.category or 'None specified'}
            - Required Skills: {', '.join(request.researchTopic.requiredSkills) if request.researchTopic.requiredSkills else 'None specified'}
            - Description: {request.researchTopic.description or 'None specified'}
            
            Provide a highly detailed analysis of how well they match.
            Return a strict JSON object with exactly these keys:
            - "strengths": (array of strings) 2 to 3 reasons why the student is a good fit.
            - "gaps": (array of strings) 1 to 2 skills or areas the student might need to learn or improve for this topic.

            Return ONLY the JSON object.
            """
            response = await gemini_model.generate_content_async(prompt)
            
            import json
            text = response.text.strip()
            data = json.loads(text)
            
            if "strengths" in data:
                strengths = data["strengths"]
            if "gaps" in data:
                gaps = data["gaps"]
                
        except Exception as e:
            gemini_error = str(e)
            print("Gemini API Error:", e)
    else:
        gemini_error = "API_KEY is not loaded properly."
    
    # Fallback
    if not strengths:
        student_skills_set = set([s.lower() for s in student_skills])
        topic_skills_set = set([s.lower() for s in topic_skills])
        
        matched_skills = student_skills_set.intersection(topic_skills_set)
        missing_skills = topic_skills_set.difference(student_skills_set)
        
        if matched_skills:
            strengths.append(f"Strong overlap in core requirements: {', '.join(matched_skills).title()}")
        if score > 30:
            strengths.append("Your general research interests align well with this topic's domain.")
            
        if missing_skills:
            gaps.append(f"Consider familiarizing yourself with: {', '.join(missing_skills).title()}")
            
        if not strengths:
            strengths.append("You have a foundational background but lack specific required skills.")
        if not gaps:
            gaps.append("Excellent! You meet all explicitly required skills.")
            
        if gemini_error:
            gaps.append(f"Gemini API Error: {gemini_error}")

    return {
        "matchScore": score,
        "strengths": strengths[:3],
        "gaps": gaps[:3]
    }

@app.post("/api/recommend")
async def recommend(request: RecommendRequest):
    student_dict = request.studentProfile.dict() if hasattr(request.studentProfile, 'dict') else request.studentProfile.model_dump()
    topics_dict = [t.dict() if hasattr(t, 'dict') else t.model_dump() for t in request.topics]
    
    top_topics = model.recommend_topics(student_dict, topics_dict)
    
    if API_KEY:
        try:
            import json
            gemini_model = genai.GenerativeModel("gemini-3.6-flash", generation_config={"response_mime_type": "application/json"})
            
            simplified_topics = [{"id": t['topicId'], "title": t.get('topicTitle', ''), "requiredSkills": t.get('topicSkills', [])} for t in top_topics]
            
            prompt = f"""
            You are an expert academic advisor.
            I have a student profile and a list of available research topics.
            Analyze the fit and return the top recommended topics for this specific student from the provided list.
            
            Student Profile:
            - Skills: {', '.join(request.studentProfile.skills) if request.studentProfile.skills else 'None specified'}
            - Research Interests: {request.studentProfile.researchInterests or 'None specified'}
            
            Available Topics:
            {json.dumps(simplified_topics, indent=2)}
            
            Return a strict JSON array containing exactly objects for the topics provided. Each object MUST have these keys:
            - "topicId": (string) The exact id of the recommended topic from the list provided.
            - "rationale": (string) A detailed 2-3 sentence explanation of WHY this topic is a great fit for their specific skills and interests.

            Return ONLY the JSON array.
            """
                
            response = await gemini_model.generate_content_async(prompt)
            text = response.text.strip()
            data = json.loads(text)
            
            # Map rationale back to the topics
            for d in data:
                for t in top_topics:
                    if str(t['topicId']) == str(d.get('topicId')):
                        t['rationale'] = d.get('rationale', "Highly recommended based on your profile.")
                
        except Exception as e:
            print("Gemini Recommend Error:", e)
            for t in top_topics:
                t['rationale'] = "This topic perfectly aligns with your research interests."
    else:
        for t in top_topics:
            t['rationale'] = "This topic perfectly aligns with your research interests."
            
    for t in top_topics:
        t.pop('topicTitle', None)
        t.pop('topicSkills', None)

    return top_topics

# ⚡ Realtime / Instant APIs (No Gemini, purely mathematical) ⚡

@app.post("/api/quick-score")
def quick_score(request: MatchRequest):
    student_skills = request.studentProfile.skills or []
    topic_skills = request.researchTopic.requiredSkills or []
    
    score = model.calculate_match_score(
        student_skills,
        request.studentProfile.researchInterests,
        request.researchTopic.title,
        topic_skills,
        request.researchTopic.description
    )
    return {"matchScore": score}

@app.post("/api/quick-recommend")
def quick_recommend(request: RecommendRequest):
    student_dict = request.studentProfile.dict() if hasattr(request.studentProfile, 'dict') else request.studentProfile.model_dump()
    topics_dict = [t.dict() if hasattr(t, 'dict') else t.model_dump() for t in request.topics]
    
    top_topics = model.recommend_topics(student_dict, topics_dict)
    
    for t in top_topics:
        t.pop('topicTitle', None)
        t.pop('topicSkills', None)
        t['rationale'] = "Match calculated instantly."
        
    return top_topics

@app.post("/api/score-all")
def score_all(request: RecommendRequest):
    student_dict = request.studentProfile.dict() if hasattr(request.studentProfile, 'dict') else request.studentProfile.model_dump()
    topics_dict = [t.dict() if hasattr(t, 'dict') else t.model_dump() for t in request.topics]
    
    student_skills = student_dict.get("skills", [])
    research_interests = student_dict.get("researchInterests", "")
    
    scores = {}
    for topic in topics_dict:
        score = model.calculate_match_score(
            student_skills, 
            research_interests, 
            topic.get("title", ""),
            topic.get("requiredSkills", []), 
            topic.get("description", "")
        )
        scores[topic.get("id")] = score
        
    return scores
