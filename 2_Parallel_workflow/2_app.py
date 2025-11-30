# https://paper-pulse-eval.lovable.app
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
import operator

# Load API key from .env
load_dotenv()

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define schema for structured output
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

# Define state
class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

# Evaluation functions
def evaluate_language(state: UPSCState):
    prompt = f"Evaluate the language quality of the following essay and provide feedback and assign a score out of 10:\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {"language_feedback": output.feedback, "individual_scores": [output.score]}

def evaluate_analysis(state: UPSCState):
    prompt = f"Evaluate the depth of analysis of the following essay and provide feedback and assign a score out of 10:\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {"analysis_feedback": output.feedback, "individual_scores": [output.score]}

def evaluate_thought(state: UPSCState):
    prompt = f"Evaluate the clarity of thought of the following essay and provide feedback and assign a score out of 10:\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {"clarity_feedback": output.feedback, "individual_scores": [output.score]}

def final_evaluation(state: UPSCState):
    # Summarized feedback
    prompt = f"""
    Based on the following feedbacks, create a summarized feedback:
    - Language feedback: {state['language_feedback']}
    - Depth of analysis: {state['analysis_feedback']}
    - Clarity of thought: {state['clarity_feedback']}
    """
    overall_feedback = model.invoke(prompt).content
    avg_score = sum(state['individual_scores']) / len(state['individual_scores'])
    return {"overall_feedback": overall_feedback, "avg_score": avg_score}

# Build workflow
graph = StateGraph(UPSCState)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Edges
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Essay Evaluator", layout="wide")
st.title("📝 Essay Evaluator")
st.write("Paste your essay below and get AI-powered evaluation with feedback & scores.")

# Essay input
essay_input = st.text_area("✍️ Enter your essay:", height=300)

if st.button("Evaluate Essay"):
    if essay_input.strip() == "":
        st.warning("Please enter an essay before evaluating.")
    else:
        with st.spinner("Evaluating your essay..."):
            result = workflow.invoke({"essay": essay_input})

        # Display results
        st.subheader("📊 Evaluation Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Language Score", result["individual_scores"][0])
        with col2:
            st.metric("Analysis Score", result["individual_scores"][1])
        with col3:
            st.metric("Clarity Score", result["individual_scores"][2])

        st.metric("⭐ Average Score", round(result["avg_score"], 2))

        st.subheader("🔎 Detailed Feedback")
        st.write(f"**Language Feedback:** {result['language_feedback']}")
        st.write(f"**Analysis Feedback:** {result['analysis_feedback']}")
        st.write(f"**Clarity Feedback:** {result['clarity_feedback']}")

        st.subheader("📌 Overall Feedback")
        st.success(result["overall_feedback"])
