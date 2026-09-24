from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_match_score(student_skills, research_interests, topic_title, topic_skills, topic_description):
    student_text = " ".join(student_skills) + " " + (research_interests or "")
    
    topic_text = f"{topic_title or ''} {' '.join(topic_skills)} {topic_description or ''}"
    
    if not student_text.strip() or not topic_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([student_text, topic_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(similarity * 100, 2)
    except Exception:
        return 0.0

def recommend_topics(student_profile, all_topics):
    student_skills = student_profile.get("skills", [])
    research_interests = student_profile.get("researchInterests", "")
    
    recommendations = []
    
    for topic in all_topics:
        score = calculate_match_score(
            student_skills, 
            research_interests, 
            topic.get("title", ""),
            topic.get("requiredSkills", []), 
            topic.get("description", "")
        )
        
        recommendations.append({
            "topicId": topic.get("id"),
            "matchPercentage": score,
            "topicTitle": topic.get("title", ""),
            "topicSkills": topic.get("requiredSkills", [])
        })
        
    recommendations.sort(key=lambda x: x["matchPercentage"], reverse=True)
    
    return recommendations[:3]
