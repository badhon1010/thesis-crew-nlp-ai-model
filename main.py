from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import model

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
